"""
The script is designed to produced MHW probability based on NMME.

The following steps are done to create the MHW prediction
- use the downloaded forecast to calculate detrended MHW
- use the climatology from `/Datasets.private/marinehw/NMME_preprocess/` (based on 1991-2020)
- use the threshold from `/Datasets.private/marinehw/NMME_preprocess/` (based on 1991-2020)
- the file are stored in `/Datasets.private/marinehw/NMME_newforecast/`
  the generated file will include all 2021 and onward MHW prediction (not just the new month)

"""
import warnings
from typing import Tuple
from datetime import date
import subprocess
import xarray as xr
import numpy as np
from dask.distributed import Client
from nmme_download import iri_nmme_models
from nmme_climo import read_nmme



def read_nmme_onlist(
        model_list: list[str],
        all_model_list: list[str],
        basedir: str,
        predir: str
) -> dict:
    """read in the NMME for MHW detection

    Parameters
    ----------
    model_list : list[str]
        list of string of the model name one want to include
        in the MHW probability calculation
    """

    # read user input
    da_nmem_list = []
    da_model_list = []
    da_climo_list = []
    for modelname in model_list:
        if modelname in all_model_list:

            # construct model list
            forecast_files = f'{basedir}{modelname}_forecast_??_??_??_??????.nc'
            climo_file = f'{predir}{modelname}_climo.nc'

            print('------------')
            print(modelname)
            print('------------')


            # lazy loading all dataset
            ds_nmme = read_nmme(
                forecast_files = forecast_files,
                model = modelname,
                chunks={'M':1,'L':1,'S':1}
            )

            da_model = ds_nmme['sst']

            # read climatology (1991-2020)
            da_ensmean_climo = xr.open_dataset(climo_file,chunks={'S':1,'L':1})['sst']

            if modelname in ['CanCM4i', 'GEM-NEMO','CanCM4i-IC3','GEM5-NEMO']:
                # the unit is in Kelvin
                da_model = da_model-273.15
                # prediction has value over land (use climo file as land mask)
                da_mask = da_ensmean_climo.isel(month=0,L=0).drop_vars(['month','L'])
                da_mask = da_mask.where(da_mask.isnull(),other=1)
                da_model = da_model*da_mask

            # calculate ensemble member in each model
            da_nmem = da_model.where(da_model.isnull(), other=1).sum(dim=['M'])
            da_nmem = da_nmem.where(da_nmem>0)

            # stored all models in one list
            da_nmem_list.append(da_nmem)           # number of ensemble member
            da_model_list.append(da_model)         # model output
            da_climo_list.append(da_ensmean_climo) # model climatology

    # combined all model into one dataset
    da_nmem_all = xr.concat(da_nmem_list,dim='model',join='outer')

    # create mask for every S, L, X, Y (if model number less than 2 will be masked)
    da_nmodel = (da_nmem_all/da_nmem_all).sum(dim='model')
    da_nmodel_mask = da_nmodel.where(da_nmodel>1)
    da_allmodel_mask = da_nmodel_mask.where(da_nmodel_mask.isnull(),other=1).compute()

    # calculate total member of all model
    da_nmem_all_out = (da_nmem_all*da_allmodel_mask).sum(dim='model').compute()

    return {
        'da_model_list':da_model_list,
        'da_climo_list':da_climo_list,
        'da_nmem_all_out':da_nmem_all_out,
        'da_allmodel_mask':da_allmodel_mask
    }

def output_format(
        ds:xr.Dataset
)->Tuple[xr.Dataset,dict]:
    """format the output dataset

    Parameters
    ----------
    ds : xr.Dataset
        the dataset one want to format

    Returns
    -------
    Tuple[xr.Dataset,dict]
        a list of with first object being the formated ds, 
        and the second object of encoding (dict) used for
        output
    """
    ds = ds.rename(
        dict(
            X='lon',
            Y='lat',
            L='lead_time',
            S='start_time'
        )
    )

    # reformat datetime object
    datetime = []
    for i,_ in enumerate(ds['start_time'].values):
        year_str = ds['start_time'].dt.year.values[i]
        mon_str = ds['start_time'].dt.month.values[i]
        datetime.append(np.datetime64(f'{year_str:04d}-{mon_str:02d}','D'))
    ds['start_time'] = datetime

    try :
        ds = ds.drop_vars('month')
    except ValueError :
        # when month variable is not there
        print('no month coord')

    encoding_list = {}
    encoding_list['mhw_probability'] = {}
    encoding_list['mhw_probability']['chunksizes'] = [1, 12, 100, 100]
    encoding_list['mhw_probability']['contiguous'] = False
    encoding_list['start_time'] = {}
    encoding_list['start_time']['chunksizes'] = [1]
    encoding_list['start_time']['contiguous'] = False
    encoding_list['lead_time'] = {}
    encoding_list['lead_time']['chunksizes'] = [12]
    encoding_list['lead_time']['contiguous'] = False
    encoding_list['lead_time']['dtype'] = 'float32'
    encoding_list['lon'] = {}
    encoding_list['lon']['contiguous'] = True
    encoding_list['lat'] = {}
    encoding_list['lat']['contiguous'] = True

    return ds, encoding_list


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    ### open local cluster
    client = Client(n_workers=2,threads_per_worker=60,processes=False)

    ###### Setting ######
    # specify date
    today = date.today()
    # setup the new output file name (same as date of download)
    dateform = today.strftime("%y_%m_%d")

    # directory where new simulation (inputs) and mhw forecast (outputs) is located
    BASEDIR = '/Datasets.private/marinehw/nmme_sst_raw/'

    # directory where sst threshold/climatology/trend (inputs) is located
    PREDIR = '/Datasets.private/marinehw/NMME_preprocess/'

    # directory where nmme mhw probability is located
    OUTDIR = '/Datasets.private/marinehw/nmme_mhw_prob/'

    # output filename date, MHW prediction generated date
    date = dateform

    # MHW threshold for prediction
    threshold = [90]

    # used model list
    model_use_list = [
        'CanCM4i-IC3',
        'GEM5-NEMO',
        'GFDL-SPEAR-regridded',
        'NASA-GEOSS2S',
        'COLA-RSMAS-CCSM4',
        'COLA-RSMAS-CESM1',
        'NCEP-CFSv2'
    ]

    dict_model = iri_nmme_models()
    avai_model_list = list(dict_model.keys())

    ################################## Main program start #####################################

    dict_da = read_nmme_onlist(model_use_list,avai_model_list,BASEDIR,PREDIR)

    # loop through all set threshold
    for m in threshold:
        da_mhw_list = []
        for nmodel,model in enumerate(model_use_list):
            if model in avai_model_list:
                threshold_file = f'{PREDIR}{model}_threshold{m}.nc'

                print('------------')
                print(model,' MHW detection...')
                print('------------')

                # read threshold (1991-2020)
                da_threshold = xr.open_dataset(
                    threshold_file,
                    chunks={'S':1,'L':1}
                )[f'threshold_{m}']

                print('calculating anomaly')
                da_anom = (
                    dict_da['da_model_list'][nmodel].groupby('S.month')
                    -dict_da['da_climo_list'][nmodel]
                )

                print('calculating MHW')
                da_mhw = da_anom.where(da_anom.groupby('S.month')>=da_threshold)
                da_mhw = (
                    da_mhw
                    .where(da_mhw.isnull(),other=1)
                    .sum(dim='M',skipna=True)
                )
                da_mhw_list.append(da_mhw)

        da_mhw_all = xr.concat(da_mhw_list,dim='model',join='outer')
        da_mhw_all_out = (da_mhw_all*dict_da['da_allmodel_mask']).sum(dim='model').compute()

        ds_mhw_prob = xr.Dataset()
        ds_mhw_prob['mhw_probability'] = (
            (da_mhw_all_out/dict_da['da_nmem_all_out'])*dict_da['da_allmodel_mask']
        )

        #### formating output
        ds_mhw_prob, encoding = output_format(ds_mhw_prob)

        print('file output')
        print(OUTDIR + f'NMME_prob{m}_{date}.nc')
        ds_mhw_prob.to_netcdf(OUTDIR + f'NMME_prob{m}_{date}.nc',encoding=encoding)

        command_line = f"ln -fs {OUTDIR}NMME_prob{m}_{date}.nc {OUTDIR}NMME_prob{m}_latest.nc"
        print(command_line)
        subprocess.call(command_line,
                            shell=True,
                            executable="/usr/bin/bash")
