# %%
"""
# NMME Temperature Observations to Avoid Loggerheads (TOTAL)
 
The script calculate the TOTAL threshold based on 
Brodie et al., [2023]

Using NMME model hindcast and forecast obtained from 
http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
with variable SST

"""


# %%
# start a local cluster
import sys
import warnings
import numpy as np
import xarray as xr
from dask.distributed import Client
from nmme_download import iri_nmme_models
from nmme_climo import read_nmme
from mhw_nmme_threshold import nmme_1mon_ts_quantile
from nmme_hci_climo_threshold import read_marine_index_mask

warnings.simplefilter("ignore")

if __name__ == "__main__":

    client = Client(n_workers=1,threads_per_worker=60,processes=False)
    print(client)
    print(client.cluster.dashboard_link)

    BASEDIR = '/Datasets.private/marinehw/nmme_sst_raw/'
    MASKDIR = '/Datasets.private/marinehw/nmme_marine_index_mask/'
    OUTPUTDIR = '/Datasets.private/marinehw/nmme_sst_stat/'

    if len(sys.argv) < 2:
        print("Usage: python nmme_climo_threshold_total.py <model name>")

    dict_model = iri_nmme_models()
    avai_model_list = list(dict_model.keys())

    # read user input
    model_list = []
    for model_name in sys.argv[1:]:
        if model_name in avai_model_list:
            model_list.append(str(model_name))
            print(f'{model_name} exist, add to process list')
        else:
            print(f'{model_name} does not exist')
            print(f'Make sure pick model from {avai_model_list}')

        # construct model list
        forecast_nmme_files = f'{BASEDIR}{model_name}_forecast_??_??_??_??????.nc'
        climo_file = f'{OUTPUTDIR}{model_name}_climo.nc'
        output_file = f'{OUTPUTDIR}{model_name}_climo_threshold_total.nc'

        # consistent climatology period
        climatology_period = [1991,2020]

        print('-------------')
        print(model_name)
        print('-------------')
        ds_nmme = read_nmme(
            forecast_files = forecast_nmme_files,
            model = model_name
        )

        ds_nmme = ds_nmme.where(
            (ds_nmme['S.year']>=climatology_period[0])&
            (ds_nmme['S.year']<=climatology_period[1]),
            drop=True
        )

        # read climatology (1991-2020)
        da_ensmean_climo = xr.open_dataset(climo_file,chunks={'S':1,'L':1})['sst']

        ds_mask = read_marine_index_mask(MASKDIR)
        da_total = ds_mask['TOTAL']*(ds_nmme['sst'].groupby('S.month')-da_ensmean_climo)
        # calculate area weight
        weights = np.cos(np.deg2rad(da_total.Y))
        weights.name = "weights"
        # calculate area weighted mean
        da_total_ts = da_total.weighted(weights).mean(dim=["X","Y"])

        # calculate the 6-month rolling mean of the previous six month
        da_total_ts_6mon = da_total_ts.rolling(L=6, center=False).mean()

        # the rolling mean of six month is put in the last month of the six
        #  shift 1 is needed to let the rolling mean represent the monthly
        #  value the is 1 month after the six month window
        da_total_ts_6mon['L'] = da_total_ts_6mon['L'].data+1.

        # calculate threshold based on quantile
        da_total_threshold = nmme_1mon_ts_quantile(
            da_total_ts_6mon,
            total_threshold=74
        )

        # da_total_threshold_git = nmme_total_quantile(
        #     da_total_ts_6mon_shift,
        #     total_threshold=75
        # )

        ds_total_ts = xr.Dataset()
        ds_total_ts['total_threshold'] = da_total_threshold

        print('file output')
        ds_total_ts.to_netcdf(output_file)
