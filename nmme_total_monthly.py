# %%
"""
# NMME Temperature Observations to Avoid Loggerheads (TOTAL)
 
The script calculate the TOTAL value from the start of the nmme 1991 based on 
Brodie et al., [2023]

Using NMME model hindcast and forecast obtained from 
http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
with variable SST

"""


# %%
# start a local cluster
import json
import subprocess
from datetime import date
import warnings
import numpy as np
import xarray as xr
from dask.distributed import Client
from nmme_download import iri_nmme_models
from nmme_hci_climo_threshold import read_marine_index_mask
from nmme_monthly_mhw import read_nmme_onlist
from nmme_hci_monthly import output_format

warnings.simplefilter("ignore")

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

    da_total_identified_list = []
    da_total_all_list = []
    da_total_ens_list = []
    for nmodel,model in enumerate(model_use_list):
        if model in avai_model_list:
            threshold_file = f'{PREDIR}{model}_climo_threshold_total.nc'

            print('------------')
            print(model,' TOTAL detection...')
            print('------------')

            # read threshold (1991-2020)
            da_threshold = xr.open_dataset(
                threshold_file,
                chunks={'S':1,'L':1}
            )['total_threshold']

            da_ssta = ds_mask['TOTAL']*(
                dict_da['da_model_list'][nmodel].groupby('S.month')
                -dict_da['da_climo_list'][nmodel]
            )
            # calculate area weight
            weights = np.cos(np.deg2rad(da_ssta.Y))
            weights.name = "weights"
            # calculate area weighted mean
            da_ssta_areamean = da_ssta.weighted(weights).mean(dim=["X","Y"])

            # calculate the 6-month rolling mean of the previous six month
            da_total = da_ssta_areamean.rolling(L=6, center=False).mean()

            # the rolling mean of six month is put in the last month of the six
            #  shift 1 is needed to let the rolling mean represent the monthly
            #  value the is 1 month after the six month window
            da_total['L'] = da_total['L'].data+1.
            da_total = da_total.persist()

            print('calculating TOTAL')
            # identify the event
            da_total_identified = da_total.where(da_total.groupby('S.month')>=da_threshold)
            da_total_identified = (
                da_total_identified
                .where(da_total_identified.isnull(),other=1)
                .sum(dim=['M'],skipna=True)
            ).compute()
            # calculate the total number of ensemble member that has value
            da_total_all = (
                da_total
                .where(da_total.isnull(),other=1)
                .sum(dim=['M'],skipna=True)
            ).compute()
            da_total_identified_list.append(da_total_identified)
            da_total_all_list.append(da_total_all)

            if ENS_OUTPUT:
                da_total_ens = da_total.compute()
                da_total_ens_list.append(da_total_ens)

            del da_total

    # total identified events higher than threshold (all ensemble and models)
    da_total_identified_concat = xr.concat(da_total_identified_list,dim='model',join='outer')
    da_total_identified_concat_summodels = (
        da_total_identified_concat.sum(dim='model',skipna=True)
    ).compute()

    # total month (all ensemble and models)
    da_total_all_concat = xr.concat(da_total_all_list,dim='model',join='outer')
    da_total_all_concat_summodels = (
        da_total_all_concat.sum(dim='model',skipna=True)
    ).compute()

    if ENS_OUTPUT:
        da_total_ens_all = xr.concat(da_total_ens_list,dim='model',join='outer')


    ds_total_prob = xr.Dataset()
    NOTES = 'TOTAL probability derived from '
    MODEL_LIST = ', '.join(model_use_list)
    ds_total_prob.attrs['title'] = f'{NOTES} {MODEL_LIST}'
    ds_total_prob.attrs['comment'] = 'Derived at NOAA Physical Science Laboratory'
    ds_total_prob.attrs['reference'] = (
        'Brodie et al., 2023, '+
        'https://doi.org/10.1038/s41467-023-43188-0'
    )
    ds_total_prob['total_probability'] = (
        da_total_identified_concat_summodels/da_total_all_concat_summodels
    )

    if ENS_OUTPUT:
        ds_total_ens = xr.Dataset()
        NOTES = 'TOTAL SSTA previous 6 month rolling mean for all ensemble member derived from '
        ds_total_ens.attrs['title'] = f'{NOTES} {MODEL_LIST}'
        ds_total_ens.attrs['comment'] = 'Derived at NOAA Physical Science Laboratory'
        ds_total_ens.attrs['reference'] = (
            'Brodie et al., 2023, '+
            'https://doi.org/10.1038/s41467-023-43188-0'
        )
        ds_total_ens['total'] = da_total_ens_all
        ds_total_ens['model'] = model_use_list
        ds_total_ens = ds_total_ens.drop_vars('month')
        filename = OUTDIR + f'nmme_total_ens_{date}.nc'
        ds_total_ens.to_netcdf(filename)

    #### formating output
    ds_total_prob, encoding = output_format(ds_total_prob)

    #### concating the data since 2021 to jun2024 with CanSIP-IC3 with the CanSIP-IC4
    # CanSIP-IC4 does not provide simulation from 2021 to Jun 2024 !!!!!
    NOTES = 'change 1990-01 to 2024-06 to CanSIP-IC3 and only use CanSIP-IC4 start from 2024-07'
    print(NOTES)
    ds_old = xr.open_dataset(OUTDIR+'nmme_total_CanSIP-IC3_frozen.nc')
    ds_total_prob['total_probability'].loc[{'start_time': slice('1990-01','2024-06')}] = (
        ds_old['total_probability']
    )
    ds_total_prob.attrs['model_use_notes'] = NOTES


    print('file output')
    filename = OUTDIR + f'nmme_total_{date}.nc'
    print(filename)
    ds_total_prob.to_netcdf(filename,encoding=encoding)

    command_line = (
        f"ln -fs {filename} {filename[:-11]}latest.nc"
    )
    print(command_line)
    subprocess.call(
        command_line,
        shell=True,
        executable="/usr/bin/bash"
    )
