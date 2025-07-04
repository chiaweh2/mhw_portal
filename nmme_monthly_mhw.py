"""
The script is designed to produced MHW probability based on NMME.

The following steps are done to create the MHW prediction
- use the downloaded forecast to calculate detrended MHW
- use the climatology from `/Datasets.private/marinehw/nmme_sst_stat/` (based on 1991-2020)
- use the threshold from `/Datasets.private/marinehw/nmme_sst_stat/` (based on 1991-2020)
- the file are stored in `/Datasets.private/marinehw/nmme_mhw_prob/`
  the generated file will include all 2021 and onward MHW prediction (not just the new month)

"""
import json
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
        predir: str,
        start_year: int = 2021,
        lazy: bool = False,
        chunks: dict = None,
) -> dict:
    """read in the NMME for MHW detection

    Parameters
    ----------
    model_list : list[str]
        list of string of the model name one want to include
        in the MHW probability calculation
    all_model_list : list[str]
        list of string of all the avialable model name 
        on prem
    basedir : str
        directory path to the raw NMME model output
    predir : str
        directory path to the NMME model statistics
        (climatology, threshold, linear trend etc.)
    start_year : int
        the year where before is cropped out for analysis. 
        Default for mhw is 2021 (before 2021 is using Mike J's file)
    lazy : boolen
        the output dictionary of dataarray are all in lazy mode (True),
        or some computed to avoid duplicate calculation (False, Default)
    

    Returns
    -------
    dict
        'da_model_list': 
            a list of xr.DataArray that includes each model output cropped to the
            desired time period
        'da_climo_list':
            a list of xr.DataArray that includes each model climo
        'da_nmem_all_out':
            a xr.DataArray represents total member of all models used
        'da_allmodel_mask':
            a xr.DataArray represents mask for every S, L, X, Y 
            (if "model" number less than 2 will be masked)
    """

    if chunks is None:
        chunks = {'M':-1,'L':-1,'S':1}

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
            # ds_nmme = read_nmme(
            #     forecast_files = forecast_files,
            #     model = modelname,
            #     chunks={'M':1,'L':1,'S':1}
            # )
            ds_nmme = read_nmme(
                forecast_files = forecast_files,
                model = modelname,
                chunks = chunks
            )


            # crop to only calculate the probability after 2020 (after Mike J's file)
            ds_nmme = ds_nmme.where(ds_nmme['S.year']>(start_year-1),drop=True)

            da_model = ds_nmme['sst']

            # read climatology (1991-2020)
            # da_ensmean_climo = xr.open_dataset(climo_file,chunks={'S':1,'L':1})['sst']
            da_ensmean_climo = xr.open_dataset(climo_file)['sst']

            # calculate ensemble member in each model
            da_nmem = da_model.where(da_model.isnull(), other=1).sum(dim=['M'])
            da_nmem = da_nmem.where(da_nmem>0)

            # stored all models in one list
            da_nmem_list.append(da_nmem)           # number of ensemble member
            da_model_list.append(da_model)         # model output
            da_climo_list.append(da_ensmean_climo) # model climatology

    # combined all model into one dataset
    if lazy :
        da_nmem_all = xr.concat(da_nmem_list,dim='model',join='outer')
    else:
        da_nmem_all = xr.concat(da_nmem_list,dim='model',join='outer').persist()

    # create mask for every S, L, X, Y (if model number less than 2 will be masked)
    da_nmodel = (da_nmem_all/da_nmem_all).sum(dim='model')
    da_nmodel_mask = da_nmodel.where(da_nmodel>1)
    if lazy :
        da_allmodel_mask = da_nmodel_mask.where(da_nmodel_mask.isnull(),other=1)
    else:
        da_allmodel_mask = da_nmodel_mask.where(da_nmodel_mask.isnull(),other=1).compute()

    # calculate total member of all model
    if lazy :
        da_nmem_all_out = (da_nmem_all*da_allmodel_mask).sum(dim='model')
    else:
        da_nmem_all_out = (da_nmem_all*da_allmodel_mask).sum(dim='model').compute()

    # release memory for the persisted da_nmem_all
    del da_nmem_all

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
        {
            'X':'lon',
            'Y':'lat',
            'L':'lead_time',
            'S':'start_time'
        }
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

    # find dimension order for chunking
    dim_list = list(ds['mhw_probability'].dims)
    chunk_dict = {
        'lead_time': 12,
        'start_time': 1,
        'lon': 100,
        'lat': 100
    }
    chunk_order = []
    for dim in dim_list:
        chunk_order.append(chunk_dict[dim])

    encoding_list = {}
    encoding_list['mhw_probability'] = {}
    encoding_list['mhw_probability']['chunksizes'] = chunk_order
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
    client = Client(processes=False)

    ###### Setting ######
    # specify date
    today = date.today()
    # setup the new output file name (same as date of download)
    dateform = today.strftime("%y_%m_%d")

    # directory where new simulation (inputs) and mhw forecast (outputs) is located
    BASEDIR = '/Datasets.private/marinehw/nmme_sst_raw/'

    # directory where sst threshold/climatology/trend (inputs) is located
    PREDIR = '/Datasets.private/marinehw/nmme_sst_stat/'

    # directory where nmme mhw probability is located
    OUTDIR = '/Datasets.private/marinehw/nmme_mhw_prob/'

    # output filename date, MHW prediction generated date
    date = dateform

    # MHW threshold for prediction
    threshold = [90]

    # used model list
    with open('model_use_list.json','r',encoding='utf-8') as f:
        json_dict = json.load(f)
    model_use_list = json_dict['model_use_list']

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
        ds_mhw_prob.attrs['model_use'] = ', '.join(model_use_list)


        #### concating the data since 2021 to jun2024 with CanSIP-IC3 with the CanSIP-IC4
        # CanSIP-IC4 does not provide simulation from 2021 to Jun 2024 !!!!!
        NOTES = 'change 2021-01 to 2024-06 to CanSIP-IC3 and only use CanSIP-IC4 start from 2024-07'
        print(NOTES)
        ds_old = xr.open_dataset(OUTDIR+f'NMME_prob{m}_CanSIP-IC3_frozen.nc')
        ds_mhw_prob['mhw_probability'].loc[{'start_time': slice('2021-01','2024-06')}] = (
            ds_old['mhw_probability']
        )
        ds_mhw_prob.attrs['model_use_notes'] = NOTES


        print('file output')
        print(OUTDIR + f'NMME_prob{m}_{date}.nc')
        ds_mhw_prob.to_netcdf(OUTDIR + f'NMME_prob{m}_{date}.nc',encoding=encoding)

        command_line = f"ln -fs {OUTDIR}NMME_prob{m}_{date}.nc {OUTDIR}NMME_prob{m}_latest.nc"
        print(command_line)
        subprocess.call(
            command_line,
            shell=True,
            executable="/usr/bin/bash"
        )

        client.close()
