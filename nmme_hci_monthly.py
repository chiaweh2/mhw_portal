# %%
"""
# NMME Habitat Compression Index (HCI)
 
The script calculate the HCI value from the start of the nmme 1991 based on 
Brodie et al., [2023]

Using NMME model hindcast and forecast obtained from 
http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
with variable SST

"""


# %%
# start a local cluster
from typing import Tuple
import json
import subprocess
from datetime import date
import warnings
import numpy as np
import xarray as xr
# import dask
from dask.distributed import Client
from nmme_download import iri_nmme_models
from nmme_hci_climo_threshold import read_marine_index_mask
from nmme_monthly_mhw import read_nmme_onlist

warnings.simplefilter("ignore")

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
            'L':'lead_time',
            'S':'start_time'
        }
    )

    coords = list(ds.coords)
    variables = list(ds.variables)
    varname = []
    for var in variables:
        if var not in coords:
            varname.append(var)
    if len(varname) == 1:
        var = varname[0]
    else:
        raise NameError('More then one variables detected in the dataset')

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
    encoding_list[var] = {}
    encoding_list[var]['chunksizes'] = [1, 12]
    encoding_list[var]['contiguous'] = False
    encoding_list['start_time'] = {}
    encoding_list['start_time']['chunksizes'] = [1]
    encoding_list['start_time']['contiguous'] = False
    encoding_list['lead_time'] = {}
    encoding_list['lead_time']['chunksizes'] = [12]
    encoding_list['lead_time']['contiguous'] = False
    encoding_list['lead_time']['dtype'] = 'float32'

    return ds, encoding_list

# @dask.delayed
# def delayed_cropped(
#     year:int,
#     da_lazy:xr.DataArray,
#     da_mask:xr.DataArray,
#     cropped_dim: str='S'
# )->xr.DataArray:
#     """delayed cropping and masking of original NMME data
#     designed to introduce dask parallelizations

#     Parameters
#     ----------
#     year : int
#         the year to crop on the data
#     da_lazy : xr.DataArray
#         the entire nmme data (single model) set that need to be cropped on year
#     da_mask : xr.DataArray
#         mask to masked and reduce the data to subset not global
#     cropped_dim : str, optional
#         The year on which dimension to be cropping on, by default 'S'

#     Returns
#     -------
#     xr.DataArray
#         delayed dask array that need to be submit to processers
#     """
#     da_sst = da_mask*da_lazy.sel({cropped_dim:f'{year}'})
#     return da_sst

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    ### open local cluster
    client = Client(processes=False)
    print(client)
    print(client.cluster.dashboard_link)

    ###### Setting ######
    ENS_OUTPUT = True

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

    # directory where the marine index mask is located
    MASKDIR = '/Datasets.private/marinehw/nmme_marine_index_mask/'

    # output filename date, MHW prediction generated date
    date = dateform

    # used model list
    with open('model_use_list.json','r',encoding='utf-8') as f:
        json_dict = json.load(f)
    model_use_list = json_dict['model_use_list']

    dict_model = iri_nmme_models()
    avai_model_list = list(dict_model.keys())

    ################################## Main program start #####################################

    dict_da = read_nmme_onlist(
        model_use_list,
        avai_model_list,
        BASEDIR,
        PREDIR,
        start_year=1991,
        lazy=False,
        chunks = {'M':-1,'L':-1,'S':-1}
    )

    ds_mask = read_marine_index_mask(MASKDIR)

    da_hci_list = []
    da_hci_ens_list= []
    for nmodel,model in enumerate(model_use_list):
        if model in avai_model_list:
            threshold_file = f'{PREDIR}{model}_climo_threshold_hci.nc'

            print('------------')
            print(model,' HCI detection...')
            print('------------')

            # read threshold (1991-2020)
            da_threshold = xr.open_dataset(
                threshold_file,
                chunks={'S':1,'L':1}
            )['hci_threshold']

            ## try to parallelize the computation to each year each chunk (not necessary)
            ##   since it is already a dask lazy array chunks are paralleled automatically
            ##   delayed operation is used on python operation that does not involve dask array
            # year_list = np.unique(dict_da['da_model_list'][nmodel]['S.year'].data)
            # if len(year_list)>1:
            #     results = []
            #     for year in year_list:
            #         da_sst_delayed = delayed_cropped(
            #             year,
            #             dict_da['da_model_list'][nmodel],
            #             ds_mask['HCI_150km'],
            #             cropped_dim = 'S'
            #         )
            #         results.append(da_sst_delayed)
            #     results_computed = dask.compute(*results)
            #     da_sst = xr.concat(results_computed,dim='S')
            # else:
            #     da_sst = ds_mask['HCI_150km']*dict_da['da_model_list'][nmodel]
      
            # called persist because da_sst is used twice in the following operation
            da_sst = (ds_mask['HCI_150km']*dict_da['da_model_list'][nmodel]).persist()
            
            print('calculating HCI')
            da_hci = da_sst.where(da_sst.groupby('S.month')<=da_threshold)
            # release the da_sst from the memory
            del da_sst
            da_hci_id = (
                da_hci
                .where(da_hci.isnull(),other=1)
                .sum(dim=['M','X','Y'],skipna=True)
            ).compute()
            da_hci_list.append(da_hci_id)

            if ENS_OUTPUT:
                da_hci_ens = (
                    da_hci
                    .where(da_hci.isnull(),other=1)
                    .sum(dim=['X','Y'],skipna=True)
                ).compute()
                da_hci_ens_list.append(da_hci_ens)          


    # total grid points lower than threshold (all ensemble and models)
    da_hci_all = xr.concat(da_hci_list,dim='model',join='outer')
    da_hci_all_out = da_hci_all.sum(dim='model',skipna=True)
    if ENS_OUTPUT:
        da_hci_ens_all = xr.concat(da_hci_ens_list,dim='model',join='outer')

    # calculate total grid points in the 150km region in all multi-model-ensemble
    da_total_grids = (
        (
            ds_mask['HCI_150km']
            *dict_da['da_allmodel_mask']
            *dict_da['da_nmem_all_out']
        )
        .sum(dim=['X','Y'],skipna=True)
    )


    ds_hci_ratio = xr.Dataset()
    notes = 'HCI derived from '
    ds_hci_ratio.attrs['title'] = [f'{notes} {model}' for model in model_use_list]
    ds_hci_ratio.attrs['comment'] = 'Derived at NOAA Physical Science Laboratory'
    ds_hci_ratio.attrs['reference'] = 'Brodie et al., 2023, https://doi.org/10.1038/s41467-023-43188-0'
    ds_hci_ratio['hci'] = da_hci_all_out/da_total_grids

    if ENS_OUTPUT:
        da_total_grids_ens = (
            (
                ds_mask['HCI_150km']
                *dict_da['da_allmodel_mask']
            )
            .sum(dim=['X','Y'],skipna=True)
        )
        ds_hci_ens_ratio = xr.Dataset()
        notes = 'HCI for all ensemble member derived from '
        
        ds_hci_ens_ratio.attrs['title'] = [f'{notes} {model}' for model in model_use_list]
        ds_hci_ens_ratio.attrs['comment'] = 'Derived at NOAA Physical Science Laboratory'
        ds_hci_ens_ratio.attrs['reference'] = 'Brodie et al., 2023, https://doi.org/10.1038/s41467-023-43188-0'
        ds_hci_ens_ratio['hci'] = da_hci_ens_all/da_total_grids_ens
        ds_hci_ens_ratio['model'] = model_use_list
        ds_hci_ens_ratio = ds_hci_ens_ratio.drop_vars('month')
        filename = OUTDIR + f'nmme_hci_ens_{date}.nc'
        ds_hci_ens_ratio.to_netcdf(filename)

    #### formating output
    ds_hci_ratio, encoding = output_format(ds_hci_ratio)

    print('file output')
    filename = OUTDIR + f'nmme_hci_{date}.nc'
    print(filename)
    ds_hci_ratio.to_netcdf(filename,encoding=encoding)

    command_line = (
        f"ln -fs {filename} {filename[:-11]}latest.nc"
    )
    print(command_line)
    subprocess.call(
        command_line,
        shell=True,
        executable="/usr/bin/bash"
    )
