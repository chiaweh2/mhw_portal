# %%
"""
# NMME Marine Heat Wave
 
The script calculate the marine heat wave event based on 
Mike Jacox et al., [2022]

Using NMME model hindcast and forecast obtained from
http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
with variable SST

The steps are described below.
------------------------------
1. calculate monthly ensemble mean climatology for each model initialization month and lead time
2. calculate the anomaly for each model by subtracting the climatology
3. calculate the threshold based on the anomaly (fixed at 90, 95, 99 quantile for now)
4. calculate the events that are higher than the threshold

"""


# %%
# start a local cluster
import sys
import warnings
from dask.distributed import Client
from nmme_download import iri_nmme_models
from nmme_mhw_detrend import cal_mhw, cal_anom

warnings.simplefilter("ignore")

# %%
if __name__ == "__main__" :

    client = Client(processes=False)
    print(client)
    print(client.cluster.dashboard_link)

    BASEDIR='/Datasets.private/marinehw/nmme_sst_raw/'
    PROCDIR='/Datasets.private/marinehw/NMME_preprocess/'
    if len(sys.argv) < 2:
        print("Usage: python nmme_mhw.py <model name>")

    dict_model = iri_nmme_models()
    avai_model_list = list(dict_model.keys())

    # read user input
    model_list = []
    for modelname in sys.argv[1:]:
        if modelname in avai_model_list:
            model_list.append(str(modelname))
            print(f'{modelname} exist, add to process list')
        else:
            print(f'{modelname} does not exist')
            print(f'Make sure pick model from {avai_model_list}')

        # construct model list
        forecast_files = f'{BASEDIR}{modelname}_forecast_??_??_??_??????.nc'
        climo_file = f'{PROCDIR}{modelname}_climo.nc'
        output_file1 = f'{PROCDIR}{modelname}_threshold.nc'
        output_file2 = f'{PROCDIR}{modelname}_mhw.nc'

        # consistent threshold period
        START_YEAR = 1991
        END_YEAR = 2020

        # three fixed threshold output
        mhw_threshold = [90,95,99]

        print('-------------')
        print(modelname)
        print('-------------')

        da_anom = cal_anom(
            forecast_files,
            modelname,
            START_YEAR,
            END_YEAR,
            remove_trend=False,
            climo_file=climo_file
        )

        for m in mhw_threshold:
            output1 = output_file1[:-3]+f'{m}.nc'
            output2 = output_file2[:-3]+f'{m}.nc'
            cal_mhw(
                da_anom,
                m,
                output1,
                output2
            )
            print(output1)
            print(output2)
