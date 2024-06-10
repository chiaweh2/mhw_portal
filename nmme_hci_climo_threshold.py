# %%
"""
# NMME Habitat Compression Index (HCI)
 
The script calculate the HCI threshold based on 
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
from nmme_climo import nmme_ens_climo,read_nmme

warnings.simplefilter("ignore")

def read_marine_index_mask(mask_dir:str)->xr.Dataset:
    """read mask file into xr.Dataset

    Parameters
    ----------
    mask_dir : str
        directory string to the mask file

    Returns
    -------
    xr.Dataset
        The mask stored in the xr.Dataset
    """
    return xr.open_dataset(f'{mask_dir}hci_total_mask.nc')



if __name__ == "__main__":

    client = Client(n_workers=1,threads_per_worker=60,processes=False)
    print(client)
    print(client.cluster.dashboard_link)

    BASEDIR = '/Datasets.private/marinehw/nmme_sst_raw/'
    MASKDIR = '/Datasets.private/marinehw/nmme_marine_index_mask/'
    OUTPUTDIR = '/Datasets.private/marinehw/nmme_sst_stat/'

    if len(sys.argv) < 2:
        print("Usage: python nmme_climo_threshold_hci.py <model name>")

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
        output_file = f'{OUTPUTDIR}{model_name}_climo_threshold_hci.nc'

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

        ds_mask = read_marine_index_mask(MASKDIR)
        da_hci = ds_mask['HCI_75km']*ds_nmme['sst']
        # calculate area weight
        weights = np.cos(np.deg2rad(da_hci.Y))
        weights.name = "weights"
        # calculate area weighted mean
        da_hci_ts = da_hci.weighted(weights).mean(dim=["X","Y"])

        ds_hci_ts = xr.Dataset()
        ds_hci_ts['hci_threshold'] = da_hci_ts
        
        print('calculating climatology')
        ds_ensmean_climo = nmme_ens_climo(ds_hci_ts,climo_dim='S',ens_dim='M').compute()

        print('file output')
        ds_ensmean_climo.to_netcdf(output_file)
