"""
# NMME area percentage climatology
 
The script calculate the marine heawave area coverage 

Using NMME model hindcast and forecast obtained from 
http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
with variable SST

The steps are described below.
------------------------------
1. Using monthly ensemble mean climatology for each model initialization month and lead time 
   to determine grid point available at each S and L for global mask
2. Using each ensemble member with identified MHW as a mask of area coverage (each member)
3. Calculate the area of mhw (step2) and global/basin (step1)
4. Calculate the climatology of the area percentage based on the same period 
   (climatology_period = [1991,2020])
"""
import json
import numpy as np
import xarray as xr
from dask.distributed import Client
from nmme_download import iri_nmme_models
from nmme_areaperc_climo import read_nmme_onlist_areaperc

if __name__ == "__main__" :

    ###### Setting ######
    # directory where past simulation (inputs) is located
    MODELDIR = '/Datasets.private/marinehw/nmme_sst_raw/'

    # directory where sst climatology/trend/threshold (inputs) is located
    PREPROCDIR = '/Datasets.private/marinehw/nmme_sst_stat/'

    # directory where MHW probability is located
    PROBDIR = '/Datasets.private/marinehw/'

    # MHW threshold for prediction
    threshold = [90]

    ### open local cluster
    client = Client(processes=False)

    # used model list
    with open('model_use_list.json','r',encoding='utf-8') as f:
        json_dict = json.load(f)
    model_use_list = json_dict['model_use_list']

    dict_model = iri_nmme_models()
    avai_model_list = list(dict_model.keys())

    ################################## Main program start #####################################
    dict_da = read_nmme_onlist_areaperc(
        model_use_list,
        avai_model_list,
        MODELDIR,
        PREPROCDIR,
        region='eez'
    )

    # create Dataset to store climo area percentage
    ds_climo_areaperc = xr.Dataset()

    # calculate area weight
    weights = np.cos(np.deg2rad(dict_da['da_global'].Y))
    weights.name = "weights"

    output_files = [
        f'{PREPROCDIR}NMME_MHW_maskEneMean_areaPercEEZ_climo_1991_2020_',
        f'{PREPROCDIR}NMME_MHW_maskEneMean_areaPercEEZ_climo_detrend_1991_2020_']
    for nmodel, model in enumerate(model_use_list):
        if model != model_use_list[-1]:
            output_files[0] = output_files[0]+model+'_'
            output_files[1] = output_files[1]+model+'_'
        else:
            output_files[0] = output_files[0]+model+'.nc'
            output_files[1] = output_files[1]+model+'.nc'

    da_mhw_mask_list = [
        dict_da['da_mhw_all'],
        dict_da['da_mhw_detrend_all']
    ]
    for nout,output in enumerate(output_files):
        # calculate mhe area percentage for all models, S, L, M
        da_mhw_ts = da_mhw_mask_list[nout].weighted(weights).sum(dim=["X","Y"])
        da_mhw_climo = da_mhw_ts.groupby('S.month').mean(dim='S')

        da_global_ts = dict_da['da_global'].weighted(weights).sum(dim=["X","Y"])
        da_global_climo = da_global_ts.groupby('S.month').mean(dim='S')

        da_climo_areaperc = (da_mhw_climo/da_global_climo).compute()
        ds_climo_areaperc['eez'] = da_climo_areaperc

        ds_climo_areaperc.to_netcdf(output_files[nout])
