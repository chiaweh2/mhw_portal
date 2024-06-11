"""
The script is designed to produced detrended MHW probability based on NMME.

The following steps are done to create the MHW prediction
- use the downloaded forecast to calculate detrended MHW
- use the climatology from `/Datasets.private/marinehw/nmme_sst_stat/` (based on 1991-2020)
- use the threshold from `/Datasets.private/marinehw/nmme_sst_stat/` (based on 1991-2020)
- use the linear trend from `/Datasets.private/marinehw/nmme_sst_stat/` (based on 1991-2020)
- the file are stored in `/Datasets.private/marinehw/nmme_mhw_prob/`
  the generated file will include all 2021 and onward MHW prediction (not just the new month)

"""
import json
import warnings
import subprocess
from datetime import date
import xarray as xr
from dask.distributed import Client
from nmme_download import iri_nmme_models
from nmme_monthly_mhw import read_nmme_onlist, output_format


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
                threshold_file = f'{PREDIR}{model}_threshold_detrend{m}.nc'
                polyfit_file = f'{PREDIR}{model}_polyfit_p.nc'

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

                print('detrending')
                ds_p = xr.open_dataset(polyfit_file)
                ds_p = ds_p.mean(dim='M')
                fit = xr.polyval(da_anom['S'], ds_p.polyfit_coefficients)
                da_detrend = da_anom - fit

                print('calculating MHW')
                da_mhw = da_detrend.where(da_detrend.groupby('S.month')>=da_threshold)
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
        print(OUTDIR + f'NMME_prob{m}_detrend_{date}.nc')
        ds_mhw_prob.to_netcdf(OUTDIR + f'NMME_prob{m}_detrend_{date}.nc',encoding=encoding)

        command_line = (
            f"ln -fs {OUTDIR}NMME_prob{m}_detrend_{date}.nc {OUTDIR}NMME_prob{m}_detrend_latest.nc"
        )
        print(command_line)
        subprocess.call(
            command_line,
            shell=True,
            executable="/usr/bin/bash"
        )

        client.close()
