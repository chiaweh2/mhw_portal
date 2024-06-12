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
import xesmf as xe
import pandas as pd
from dask.distributed import Client
from shapely.geometry import Polygon,Point
from nmme_download import iri_nmme_models
from nmme_climo import read_nmme
# from nmme_monthly_mhw import read_nmme_onlist, output_format

def woa09_ocean_mask(da_var, basinname = 'pacific', var_x='lon', var_y='lat'):
    '''
    This is a function to interpolate the ocean mask (IRI provided)
    '/home/chsu/data/nodc_woa09_oceanmask.nc'
    to the desired grid based on the supplied `da_var` grid

    Parameters
    ----------
    da_var : xr.DataArray 
        The xr.DataArray provided the grid one want the mask to 
        be regridded to.
    basinname : string
		The options for this keyword argument are ['pacific',
        'atlantic','indian'].
    var_x : string
		The name of the x coordinate in da_var since for some model 
        grid would name the var_x as x which xesmf cannot understand. 
    var_y : string
		The name of the y coordinate in da_var since for some model 
        grid would name the var_y as y which xesmf cannot understand. 

    Returns
    -------
    da_omask_regrid : xr.DataArray
        The regridded ocean basin mask that fits the da_var grids.
        The mask should be able to be applied directly on the da_var.

    Raises
    ------
    
    '''

    # open original ocean mask from NODC WOA09
    input_file = '/home/chsu/data/nodc_woa09_oceanmask.nc'
    ds_omask = xr.open_dataset(input_file)

    # taking the surface mask (other levels are possible but need feature implementation)
    da_mask_lev0 = ds_omask.basin.isel(Z=0,drop=True)

    # determine the basin mask (other basins are possible but need feature implementation)
    if basinname == 'pacific':
        da_mask_lev0 = da_mask_lev0.where(da_mask_lev0==2)
    elif basinname == 'atlantic':
        da_mask_lev0 = da_mask_lev0.where(da_mask_lev0==1)
    elif basinname == 'indian':
        da_mask_lev0 = da_mask_lev0.where(da_mask_lev0==3)

    # convert to 1 in the basin
    da_mask_lev0 = da_mask_lev0/da_mask_lev0

    # rename X, Y to lon, lat (hard coded due to the mask file
    # has this naming.)
    da_mask_lev0 = da_mask_lev0.rename({'X':'lon','Y':'lat'})

    # rename data coordinate
    da_var = da_var.rename({var_x:'lon',var_y:'lat'})

    # xr.Dataset is needed for xe.regidder
    ds_omask_temp = xr.Dataset()
    ds_omask_temp['basin'] = da_mask_lev0
    ds_var_temp = xr.Dataset()
    ds_var_temp['var'] = da_var

    regridder_mask = xe.Regridder(ds_omask_temp,
                                  ds_var_temp,
                                  'bilinear',
                                  periodic=True)

    da_omask_regrid = regridder_mask(da_mask_lev0)

    da_omask_regrid = da_omask_regrid.rename({'lon':var_x,'lat':var_y})

    return da_omask_regrid

def read_nmme_onlist_areaperc(
        model_list: list[str],
        all_model_list: list[str],
        basedir: str,
        predir: str,
        region: str = 'global'
) -> dict:
    """read in the NMME for MHW area percentage count

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
    region : str
        region for the global ocean mask. Default is 
        based on the original NMME model ocean point,
        with defualt value of 'global'. Set to 'eez'
        for west coast california only mask

    Returns
    -------
    dict
        'da_mhw_all':
            a xr.DataArray represents the MHW detect 
            used for global area percentage count
        'da_mhw_detrend_all':
            a xr.DataArray represents the MHW detect 
            detrended used for global area percentage count
        'da_global':
            a xr.DataArray represents mask for every S, L, X, Y 
            (if "model" number less than 2 will be masked)

    """
    da_nmem_list = []
    da_model_list = []
    da_climo_list = []
    da_mhw_list = []
    da_mhw_detrend_list = []
    for modelname in model_list:
        if modelname in all_model_list:

            # construct model list
            forecast_files = f'{basedir}{modelname}_forecast_??_??_??_??????.nc'
            climo_file = f'{predir}{modelname}_climo.nc'
            mhw_file = f'{predir}{modelname}_mhw90.nc'
            mhw_detrend_file = f'{predir}{modelname}_mhw_detrend90.nc'

            print('------------')
            print(modelname)
            print('------------')


            # lazy loading all dataset
            ds_nmme = read_nmme(
                forecast_files = forecast_files,
                model = modelname,
                chunks={'M':1,'L':1,'S':1}
            )

            # crop 1991-2020
            ds_nmme = ds_nmme.where(
                (ds_nmme['S.year']>=1991)&
                (ds_nmme['S.year']<=2020),
                drop=True
            )
            da_model = ds_nmme['sst']

            # read climatology (1991-2020)
            da_ensmean_climo = xr.open_dataset(climo_file,chunks={'S':1,'L':1})['sst']

            # read mhw (1991-2020)
            da_mhw = xr.open_dataset(mhw_file,chunks={'S':1,'L':1})['mhw_90']
            da_mhw_detrend = xr.open_dataset(mhw_detrend_file,chunks={'S':1,'L':1})['mhw_90']

            # calculate ensemble member in each model
            da_nmem = da_model.where(da_model.isnull(), other=1).sum(dim=['M'])
            da_nmem = da_nmem.where(da_nmem>0)

            # stored all models in one list
            da_nmem_list.append(da_nmem)           # number of ensemble member
            da_model_list.append(da_model)         # model output
            da_climo_list.append(da_ensmean_climo) # model climatology
            da_mhw_list.append(da_mhw)                    # model mhw
            da_mhw_detrend_list.append(da_mhw_detrend)    # model mhw

    # combined all model into one dataset
    da_nmem_all = xr.concat(da_nmem_list,dim='model',join='outer')
    da_mhw_all = xr.concat(da_mhw_list,dim='model',join='outer')
    da_mhw_detrend_all = xr.concat(da_mhw_detrend_list,dim='model',join='outer')

    # create mask for every S, L, X, Y (if model number less than 2 will be masked)
    da_nmodel = (da_nmem_all/da_nmem_all).sum(dim='model')
    da_nmodel_mask = da_nmodel.where(da_nmodel>1)
    da_allmodel_mask = da_nmodel_mask.where(da_nmodel_mask.isnull(),other=1).compute()

    # calculate total member of all model
    da_nmem_all_out = (da_nmem_all*da_allmodel_mask).sum(dim='model').compute()

    # calculate global ocean mask based on all models, S, L
    #  (with minimum 2 models criteria included)
    da_global = da_nmem_all_out.where(da_nmem_all_out==0,other=1.)
    if region == 'eez':
        # read region
        x_list_mj, y_list_mj = read_eez()
        # df_eez=pd.read_csv('/home/chsu/mhw_portal/resource/west_coast_eez_outline.csv',header=None)
        # y_list_mj = df_eez[0].values
        # x_list_mj = df_eez[1].values

        # # change -180-180 to 0-360
        # x_list_mj_shift = np.copy(x_list_mj)
        # x_ind = np.where(x_list_mj<0.)
        # x_list_mj_shift[x_ind] = x_list_mj[x_ind]+360.

        # polygon_coords = [(x_list_mj[n],y_list_mj[n]) for n in range(len(x_list_mj))]
        # region_polygon = Polygon(polygon_coords)

        da_global_region = region_mask(x_list_mj,y_list_mj,da_global,xname='X',yname='Y')
        da_global = da_global*da_global_region
        # if da_global['X'].min().data<0:
        #     # change mask coordinate to -180-180
        #     da_global_region['X'] = da_global_region['X'].where(
        #         da_global_region['X']<=180.,
        #         other=da_global_region['X']-360.
        #     )


    # calculate mhw mask based on all models, S, L, M
    da_mhw_all = da_mhw_all.where(da_mhw_all.isnull(), other=1)
    da_mhw_all = da_mhw_all*da_global

    da_mhw_detrend_all = da_mhw_detrend_all.where(da_mhw_detrend_all.isnull(), other=1)
    da_mhw_detrend_all = da_mhw_detrend_all*da_global

    return {
        'da_mhw_all':da_mhw_all,
        'da_mhw_detrend_all':da_mhw_detrend_all,
        'da_global':da_global
    }

def read_eez():
    """
    The function read the EEZ csv file.
    """

    df_eez=pd.read_csv('/home/chsu/mhw_portal/resource/west_coast_eez_outline.csv',header=None)
    y_list = df_eez[0].values
    x_list = df_eez[1].values

    return x_list,y_list


def region_mask(
        region_lon: list[float],
        region_lat: list[float],
        da_data: xr.DataArray,
        xname='lon',
        yname='lat'
) -> xr.DataArray:
    """generate regional mask based on shapefile-like coordinate list

    Parameters
    ----------
    region_lon : list
        longitude coordinate list in float that 
        indicate the outline of the mask
    region_lat : list
        latitude coordinate list in float that 
        indicate the outline of the mask
    da_data : xr.DataArray
        the data which one want the mask to be
        applied on. The data array is in the input
        to provide the xr.DataArray strucuter and
        longitude latitude info. Data itself will
        not be used.
    xname : str, optional
        longitude name of the dataset, by default 'lon'
    yname : str, optional
        latitude name of the dataset, by default 'lat'

    Returns
    -------
    xr.DataArray
        The mask based on the lon, lat list in the 
        data strucuture and information provided by 
        the `da_data`
    """

    # .copy() to avoid changing the da_data original lon lat 
    lon_array = da_data[xname].data.copy()
    lat_array = da_data[yname].data.copy()

    # if data is 0-360 change to -180-180 to match shape file format
    lon_array[np.where(lon_array>180.)[0]] = lon_array[np.where(lon_array>180.)[0]]-360.
    lon_array = np.sort(lon_array)

    mask = np.zeros([len(lat_array),len(lon_array)])+np.nan

    da_mask = xr.DataArray(
                data=mask,
                dims=[yname,xname],
                coords={
                    xname:lon_array,
                    yname:lat_array
                }
    )

    polygon_coords = [(region_lon[n],region_lat[n]) for n in range(len(region_lon))]
    region_polygon = Polygon(polygon_coords)

    for lon in lon_array:
        for lat in lat_array:
            point = Point(lon, lat)
            if region_polygon.contains(point):
                ii = np.where(lon_array==lon)[0]
                jj = np.where(lat_array==lat)[0]
                da_mask[jj,ii] = 1

    # change mask coordinate to 0-360
    da_mask[xname] = da_mask[xname].where(da_mask[xname]>0., other=da_mask[xname]+360.)
    da_mask = da_mask.sortby(da_mask[xname])

    return da_mask

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
    dict_da = read_nmme_onlist_areaperc(model_use_list,avai_model_list,MODELDIR,PREPROCDIR)

    # create Dataset to store climo area percentage
    ds_climo_areaperc = xr.Dataset()

    # calculate area weight
    weights = np.cos(np.deg2rad(dict_da['da_global'].Y))
    weights.name = "weights"

    output_files = [
        f'{PREPROCDIR}NMME_MHW_maskEneMean_areaPerc_climo_1991_2020_',
        f'{PREPROCDIR}NMME_MHW_maskEneMean_areaPerc_climo_detrend_1991_2020_']
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
        ds_climo_areaperc['global'] = da_climo_areaperc

        # calculate basin masks
        basin_names = ['pacific','atlantic','indian']
        for basin in basin_names:
            da_basinmask = woa09_ocean_mask(
                dict_da['da_global'],
                basinname = basin,
                var_x='X',
                var_y='Y'
            )

            da_mhw_ts = (da_mhw_mask_list[nout]*da_basinmask).weighted(weights).sum(dim=["X","Y"])
            da_mhw_climo = da_mhw_ts.groupby('S.month').mean(dim='S')

            da_global_ts = (dict_da['da_global']*da_basinmask).weighted(weights).sum(dim=["X","Y"])
            da_global_climo = da_global_ts.groupby('S.month').mean(dim='S')

            da_climo_areaperc = (da_mhw_climo/da_global_climo).compute()
            ds_climo_areaperc[basin] = da_climo_areaperc

        ds_climo_areaperc.to_netcdf(output_files[nout])
