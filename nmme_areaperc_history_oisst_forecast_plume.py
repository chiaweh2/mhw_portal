"""
The script is for creating the 
1. Historical time series combine with the forecast MMM plot
2. limited historical time series with the forecast plume plot

"""
import warnings
from datetime import date,datetime,timedelta
import cftime
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from dask.distributed import Client
from nmme_climo import read_nmme, iri_nmme_models

warnings.filterwarnings("ignore")

##################################     Function      ######################################
def find_rank(arr):
    """
    This find the ranking of the last element value in an array 

    Parameters
    ----------
    arr : numpy array 
        The array (i.e. time series) of the last element which user
        like to know the ranking of the value in the entire array
    Returns
    -------
    ranking : integer
        The ranking of the value of the last element in the entire
        array.

    Raises
    ------
    """

    # Sort the array and get the indices of the sorted elements
    sorted_indices = np.argsort(arr)

    # Find the index of the last element in the sorted array
    last_element_index = np.where(sorted_indices == len(arr)-1)[0][0]

    return len(arr) - last_element_index

def all_obs_area_mask(proc_time='latest'):
    """
    calculate MHW mask based on OISST 
    (whole time series regridded to NMME monthly mean)
    Dataset used:
    - oisst anom trend/detrended
    - oisst threshold trend/detrended quantile 90

    Returns
    -------
    The following is returned in one dictionary object with
    key-value pair (key name shown below)
    mhw_trend : xr.DataArray type
        the mask based on MHW with trend
    mhw_detrended : xr.DataArray type
        the mask based on MHW detrended
    global : xr.DataArray type
        the mask based on MHW threshold to demostrate the global
        ocean area mask in the OISST dataset

    """
    # read in the observational data trend/detrended
    OBSBASEDIR = '/Datasets.private/marinehw/'
    ds_obs = xr.open_dataset(
        f'{OBSBASEDIR}oisst.mon.anom_{proc_time}.nc',use_cftime=True
    )
    ds_obs_threshold = xr.open_dataset(
        f'{OBSBASEDIR}oisst.mon.quantile90_{proc_time}.nc',use_cftime=True
    )

    ds_obs_detrend = xr.open_dataset(
        f'{OBSBASEDIR}oisst.mon.detrend.anom_{proc_time}.nc',use_cftime=True
    )
    ds_obs_threshold_detrend = xr.open_dataset(
        f'{OBSBASEDIR}oisst.mon.detrend.quantile90_{proc_time}.nc',use_cftime=True
    )

    # use obs sst anom to determine MHW mask
    da_mask = ds_obs.where(
        ds_obs.sst_anom.groupby('time.month')>ds_obs_threshold
    )['sst_anom']
    da_mask = da_mask.where(da_mask.isnull(),other=1)

    da_mask_detrend = ds_obs_detrend.where(
        ds_obs_detrend.sst_anom.groupby('time.month')>ds_obs_threshold_detrend
    )['sst_anom']
    da_mask_detrend = da_mask_detrend.where(da_mask_detrend.isnull(),other=1)

    # use MHW threshold to determine global mask
    da_global_mask = ds_obs_threshold.sst_anom.isel(month=0)
    da_global_mask = da_global_mask.where(da_global_mask.isnull(),other=1)

    return {
        'mhw_trend':da_mask,
        'mhw_detrended':da_mask_detrend,
        'global':da_global_mask
    }

def nmme_newest_forecast(threshold=90):
    """
    calculate MHW mask based on NMME newest 12 month forecast 
    Dataset used:
    - newforecast from NMME 
    - NMME climatology (1991-2020)
    - NMME threshold (1991-2020)
    - NMME linear trend coeff. (1991-2020)

    Returns
    -------
    The following is returned in one dictionary object with
    key-value pair (key name shown below)
    mhw_trend : xr.DataArray type
        the mask based on MHW with trend
    mhw_detrended : xr.DataArray type
        the mask based on MHW detrended
    global : xr.DataArray type
        the mask based on MHW threshold to demostrate the global
        ocean area mask in the OISST dataset

    """
    ##### Setting ######
    # directory where new simulation (inputs) and mhw forecast (outputs) is located
    basedir = '/Datasets.private/marinehw/nmme_sst_raw/'

    # directory where sst threshold/climatology/trend (inputs) and mhw hindcast (outputs) is located
    predir = '/Datasets.private/marinehw/nmme_sst_stat/'

    # Get the current date
    today = datetime.now()

    # Get year, month, and day from the current date
    cyear = today.year
    cmonth = today.month
    prev_month_datetime = today.replace(day=1)
    prev_month_datetime = prev_month_datetime - timedelta(days=1)
    pyear = prev_month_datetime.year
    pmonth = prev_month_datetime.month


    # used model list
    model_list = [
        'CanCM4i-IC3',
        'GEM5-NEMO',
        'GFDL-SPEAR-regridded',
        'NASA-GEOSS2S',
        'COLA-RSMAS-CCSM4',
        'COLA-RSMAS-CESM1',
        'NCEP-CFSv2'
    ]
    dict_model = iri_nmme_models()
    all_model_list = list(dict_model.keys())


    da_nmem_list = []
    da_model_list = []
    da_climo_list = []
    avai_model_list = []
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

            try :
                da_model = da_model.sel(
                    S=cftime.Datetime360Day(cyear, cmonth, 1, 0, 0, 0, 0, has_year_zero=True)
                )
            except KeyError:
                # if any of the date in slice does not exist, the data is fill with NaN
                da_model = da_model.isel(S=-1)*np.nan
                da_model['S'] = cftime.Datetime360Day(cyear, cmonth, 1, 0, 0, 0, 0, has_year_zero=True)

            # read climatology (1991-2020)
            da_ensmean_climo = xr.open_dataset(climo_file,chunks={'S':1,'L':1})['sst']

            # calculate ensemble member in each model
            da_nmem = da_model.where(da_model.isnull(), other=1).sum(dim=['M'])
            da_nmem = da_nmem.where(da_nmem>0)

            # stored all models in one list
            da_nmem_list.append(da_nmem)           # number of ensemble member
            da_model_list.append(da_model)         # model output
            da_climo_list.append(da_ensmean_climo) # model climatology
            avai_model_list.append(modelname)


    # combined all model into one dataset
    da_model_all = xr.concat(
        [da for da in da_model_list if type(da) != type("string")],
        dim='model',
        join='outer'
    ).compute()
    da_model_all['model'] = avai_model_list
    da_climo_all = xr.concat(
        [da for da in da_climo_list if type(da) != type("string")],
        dim='model',
        join='outer'
    ).compute()
    da_climo_all['model'] = avai_model_list
    da_nmem_all = xr.concat(
        [da for da in da_nmem_list if type(da) != type("string")],
        dim='model',
        join='outer'
    ).compute()
    da_nmem_all['model'] = avai_model_list
    da_ensmask_all = xr.concat(
        [da for da in da_model_list if type(da) != type("string")],
        dim='model',
        join='outer'
    )
    da_ensmask_all['model'] = avai_model_list

    da_ensmask_all = da_ensmask_all.sum(dim=['X','Y']).compute()
    da_ensmask_all = da_ensmask_all.where(da_ensmask_all==0,other=1)
    da_ensmask_all = da_ensmask_all.where(da_ensmask_all==1)

    # create mask for every S, L, X, Y (if model number less than 2 will be masked)
    da_nmodel = (da_nmem_all/da_nmem_all).sum(dim='model')
    da_nmodel_mask = da_nmodel.where(da_nmodel>1)
    da_allmodel_mask = da_nmodel_mask.where(da_nmodel_mask.isnull(),other=1).compute()

    # calculate total member of all model for masking
    da_nmem_all = (da_nmem_all*da_allmodel_mask).compute()
    da_nmem_all_out = da_nmem_all.sum(dim='model')
    da_global_mask = da_nmem_all_out.where(da_nmem_all_out==0,other=1.)

    # loop through all set threshold
    da_mhw_list = []
    avai_model_list = []
    m = threshold
    for modelname in model_list:
        if modelname in all_model_list:

            # construct model file
            threshold_file = f'{predir}{modelname}_threshold{m}.nc'

            print('------------')
            print(modelname,' MHW detection...')
            print('------------')

    
            # read threshold (1991-2020)
            da_threshold = xr.open_dataset(
                threshold_file,
                chunks={'S':1,'L':1}
            )[f'threshold_{m}']

            print('calculating anomaly')
            month = da_model_all.sel(model=modelname)['S.month'].data
            da_anom = da_model_all.sel(model=modelname) - da_climo_all.sel(model=modelname).sel(month=month)
            avai_model_list.append(modelname)


            print('calculating MHW')
            month = da_anom.month.data
            da_mhw_temp = da_anom.where(da_anom>=da_threshold.sel(month=month))
            da_mhw = da_mhw_temp.where(da_mhw_temp.isnull(),other=1)
            da_mhw_list.append(da_mhw)

    da_mhw_list = [da for da in da_mhw_list if ~isinstance(da, str)]
    da_mhw_all = xr.concat(da_mhw_list,dim='model',join='outer')
    da_mhw_all_out = (da_mhw_all*da_allmodel_mask).compute()
    da_mhw_all_out['model'] = avai_model_list
    da_mhw_all_out = da_mhw_all_out.where(da_mhw_all_out.isnull(),other=1)


    #### without trend
    # late two time stamp are needed for trend remove
    da_model_list = []   # previous da_model_list emptied
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

            try:
                da_model = da_model.sel(
                    S=slice(
                        cftime.Datetime360Day(pyear, pmonth, 1, 0, 0, 0, 0, has_year_zero=True),
                        cftime.Datetime360Day(cyear, cmonth, 1, 0, 0, 0, 0, has_year_zero=True)
                    )
                )
            except KeyError:
            # if any of the date in slice does not exist, the data is fill with NaN
                da_model = da_model.isel(S=range(-2,0))*np.nan
                da_model['S'] = [
                    cftime.Datetime360Day(pyear, pmonth, 1, 0, 0, 0, 0, has_year_zero=True),
                    cftime.Datetime360Day(cyear, cmonth, 1, 0, 0, 0, 0, has_year_zero=True)
                ]
            if len(da_model.S) != 2:
                 # if only one date in slice does exist, the data is still fill with NaN
                da_model = da_model.isel(S=range(-2,0))*np.nan
                da_model['S'] = [
                    cftime.Datetime360Day(pyear, pmonth, 1, 0, 0, 0, 0, has_year_zero=True),
                    cftime.Datetime360Day(cyear, cmonth, 1, 0, 0, 0, 0, has_year_zero=True)
                ]


            da_model_list.append(da_model)         # model output


    # loop through all set threshold
    da_mhw_list = []  # previous da_mhw_list emptied
    avai_model_list = [] # previous avai_model_list emptied
    for modelname in model_list:
        if modelname in all_model_list:
            # construct model file
            threshold_detrend_file = f'{predir}{modelname}_threshold_detrend{m}.nc'
            polyfit_file = f'{predir}{modelname}_polyfit_p.nc'

            print('------------')
            print(modelname,' MHW detection...')
            print('------------')

            # read threshold (1991-2020)
            da_threshold = xr.open_dataset(
                threshold_detrend_file,
                chunks={'S':1,'L':1}
            )[f'threshold_{m}']

            print('calculating anomaly and detrend')
            month = da_model_all.sel(model=modelname)['S.month'].data
            da_anom = (
                da_model_all.sel(model=modelname).groupby('S.month')
                - da_climo_all.sel(model=modelname).sel(month=month)
            )
            ds_p = xr.open_dataset(polyfit_file)
            ds_p = ds_p.mean(dim='M')
            fit = xr.polyval(da_anom['S'], ds_p.polyfit_coefficients)
            da_detrend = (da_anom.isel(S=-1) - fit.isel(S=-1)).squeeze()
            avai_model_list.append(modelname)

            print('calculating MHW')
            month = da_detrend.month.data
            da_mhw_temp = da_detrend.where(da_detrend>=da_threshold.sel(month=month))
            da_mhw = da_mhw_temp.where(da_mhw_temp.isnull(),other=1)
            da_mhw_list.append(da_mhw)

    da_mhw_list = [da for da in da_mhw_list if ~isinstance(da, str)]
    da_mhw_all = xr.concat(da_mhw_list,dim='model',join='outer')
    da_mhw_all_out_detrend = (da_mhw_all*da_allmodel_mask).compute()
    da_mhw_all_out_detrend['model'] = avai_model_list
    da_mhw_all_out_detrend = da_mhw_all_out_detrend.where(da_mhw_all_out_detrend.isnull(),other=1)

    return {
        'mhw_trend':da_mhw_all_out,
        'mhw_detrended':da_mhw_all_out_detrend,
        'global':da_global_mask,
        'ens_num':da_nmem_all,
        'ens_mask':da_ensmask_all
    }

def area_weighted_sum(da, xname='lon', yname='lat'):
    """
    Calculate the area-weighted sum of the xr.DataArray

    Parameters
    ----------
    da : xr.DataArray type 
        The data array with x(lon) and y(lat) dimension one wanted
        to do the area-weighted sum. 
    xname : string type 
        The longitude name of the data array.
    yname : string type 
        The latitude name of the data array.

    Returns
    -------
    da_aws : xr.DataArray type
        the area-weighted sum result

    """
    # calculate weight
    weights = np.cos(np.deg2rad(da[yname]))
    weights.name = "weights"

    # area weighted sum
    da_aws = (da).weighted(weights).sum(dim=[xname,yname])

    return da_aws

def plot_plume(da_nmme,da_obs,PREVMONTH=24,RANK=True):
    """
    The function to create plume plot
    """
    fig=plt.figure(1,figsize=(9,12))
    ax1=fig.add_axes([0,0.75,1,0.3])
    
    
    # colors = ['C0','C1','C2','C3','C4','C5']
    for nmodel,model in enumerate(da_nmme.model.data):
        color = f'C{nmodel}'
        for m in da_nmme.M.data:
            # plot each ensemble member
            (
            da_nmme
            .sel(model=model)
            .sel(M=int(m))
            .plot(x='valid_time',ax=ax1,alpha=0.15,color=color)
            )

        # plot continue dash
        last_obs_time = [da_obs[-1].time.dt.year.data,da_obs[-1].time.dt.month.data]
        con_time = xr.cftime_range(
            start=f'{int(last_obs_time[0])}-{int(last_obs_time[1]):02d}-01',
            periods=2,
            freq='MS'
        )
        con_data = np.array([da_obs[-1].data,da_nmme.sel(model=model).mean(dim='M')[0].data])
        ax1.plot(con_time,con_data,linestyle='dashed',color='k',linewidth=0.5)

        # plot ensemble mean of each model
        (
            da_nmme
            .sel(model=model)
            .mean(dim='M')
            .plot(
                x='valid_time',
                ax=ax1,
                color=color,
                linewidth=2,
                marker='o',
                label=model
            )
        )
    print(f'obs last month {da_obs[-1].data}')
    # plot multi-model mean of ensemble mean
    (
        da_nmme
        .mean(dim='M')
        .mean(dim='model')
        .plot(
            x='valid_time',
            ax=ax1,
            color='k',
            linewidth=2.5,
            marker='o',
            linestyle='dashed',
            label='Multi-model mean'
        )
    )
    da_nmme_mmm = da_nmme.mean(dim='M').mean(dim='model')
    print(f'forecast first month {da_nmme_mmm[0].data}')
    # plot continue dashed line
    con_data = np.array([da_obs[-1].data,da_nmme.mean(dim='M').mean(dim='model')[0].data])
    ax1.plot(con_time,con_data,linestyle='dashed',color='k',linewidth=2.5)

    # plot multi-model mean of ensemble mean lead time = 0.5
    da_point = da_nmme.mean(dim='M').mean(dim='model').isel(L=0)
    ### indigo big spot
    # ax1.plot(da_point['valid_time'].data,da_point.data,
    # color='indigo',marker='o',markersize=12,linestyle='None')
    ax1.plot(da_point['valid_time'].data,da_point.data,
    color='k',marker='o',markersize=12,linestyle='None')


    # plot observation
    da_obs[-PREVMONTH:].plot(ax=ax1,label='OISSTv2',linewidth=2,color='dimgray',marker='o')

    btime = da_obs[-PREVMONTH:].time
    ftime = da_nmme.valid_time
    tlabel = []
    tvalue = []
    for bt in btime:
        new_date = (
            date.fromisoformat(f'{bt.dt.year.data}-'+
                                f'{bt.dt.month.data:02d}-'+
                                f'{bt.dt.day.data:02d}')
        )
        tlabel.append(new_date.strftime("%b %Y"))
        tvalue.append(cftime.datetime(bt.dt.year.data,bt.dt.month.data,bt.dt.day.data))
    for ft in ftime:
        new_date = (
            date.fromisoformat(f'{ft.dt.year.data}-'+
                                f'{ft.dt.month.data:02d}-'+
                                f'{ft.dt.day.data:02d}')
        )
        tlabel.append(new_date.strftime("%b %Y"))
        tvalue.append(cftime.datetime(ft.dt.year.data,ft.dt.month.data,ft.dt.day.data))

    ax1.legend(loc='upper left',fontsize=14,frameon=False)
    ax1.set_ylabel('Ocean area percentage (%)',{'size':'15'},color='k')
    ax1.set_yticks(np.arange(0,0.61,0.1))
    ax1.set_yticklabels([f"{int(n*100)}%" for n in np.arange(0,0.61,0.1)])
    ax1.tick_params(axis='y',labelsize=12,labelcolor='k')
    ax1.set_xlabel('',{'size':'12'},color='k')
    ax1.set_xticks(tvalue[::2])
    ax1.set_xticklabels(tlabel[::2])
    ax1.tick_params(axis='x',labelsize=12,labelcolor='k',rotation=70)

    # calculate the ranking of the with trend time series
    forecast_ranking = find_rank(
        np.append(
            da_obs.data,
            da_nmme.mean(dim='M').mean(dim='model')[0].data
        )
    )

    if RANK:
        ax1.text(da_point['valid_time'].data, da_point.data+0.15,
            f'{tlabel[-12]} forecast',
            color='k',
            weight='bold',
            size=13)
        ax1.text(da_point['valid_time'].data, da_point.data+0.15-0.03,
            f'ranks {forecast_ranking} of {len(da_obs.time)+1} months',
            color='k',
            weight='bold',
            size=13)


    return fig, ax1

def plot_history_forecast(ds_nmme,da_obs,da_obs_detrend):
    """
    The function to create history and forecast plot
    """
    fig=plt.figure(1,figsize=(15,12))
    ax1=fig.add_axes([0,0.75,1,0.3])

    forecasttime = []
    forecastyear= []
    for btime in da_obs.time :
        new_date = (
            date.fromisoformat(f'{btime.dt.year.data}-'+
                            f'{btime.dt.month.data:02d}-'+
                            f'{btime.dt.day.data:02d}')
        )
        forecasttime.append(new_date.strftime("%b %Y"))
        forecastyear.append(new_date.strftime("%Y"))

    ftime = ds_nmme.valid_time
    for ft in ftime:
        new_date = (
            date.fromisoformat(f'{ft.dt.year.data}-'+
                                f'{ft.dt.month.data:02d}-'+
                                f'{ft.dt.day.data:02d}')
        )
        forecasttime.append(new_date.strftime("%b %Y"))
        forecastyear.append(new_date.strftime("%Y"))

    # time series
    ax1.plot(np.arange(len(da_obs.time)),
            da_obs,
            label='OISSTv2 with trend',
            linewidth=2,
            color='C1',
            alpha=0.6)

    ax1.plot(np.arange(len(da_obs_detrend.time)),
            da_obs_detrend,
            label='OISSTv2 without trend',
            linewidth=2,
            color='C0',
            alpha=0.6)

    # # horizontal line for the last element
    # ax1.plot(np.arange(len(da_perc_obs_ts.time)),
    #         np.ones(len(da_perc_obs_ts.time))*da_perc_obs_ts[-1].data,
    #         color='k',
    #         linestyle='--',
    #         linewidth=1)

    # ax1.plot(np.arange(len(da_perc_obs_ts_detrend.time)),
    #         np.ones(len(da_perc_obs_ts.time))*da_perc_obs_ts_detrend[-1].data,
    #         color='k',
    #         linestyle='--',
    #         linewidth=1)

    # horizontal line for the last element
    ax1.plot(np.arange(len(da_obs.time)),
            np.ones(len(da_obs.time))*ds_nmme['global'].mean(dim='M').mean(dim='model')[0].data,
            color='k',
            linestyle='--',
            linewidth=1)

    ax1.plot(
        np.arange(len(da_obs_detrend.time)),
        np.ones(
            len(da_obs_detrend.time)
        )*ds_nmme['global_detrend'].mean(dim='M').mean(dim='model')[0].data,
        color='k',
        linestyle='--',
        linewidth=1
    )


    # plot multi-model mean of ensemble mean
    lastind = len(da_obs.time)
    ax1.plot(np.arange(lastind,lastind+12),
            ds_nmme['global'].mean(dim='M').mean(dim='model'),
            color='C1',
            linestyle='dashed',
            label='forecast with trend')

    # plot multi-model mean of ensemble mean
    ax1.plot(np.arange(lastind,lastind+12),
            ds_nmme['global_detrend'].mean(dim='M').mean(dim='model'),
            color='C0',
            linestyle='dashed',
            label='forecast without trend')


    # dot for the last element
    ax1.plot(np.arange(len(da_obs.time))[-1],
            da_obs[-1],
            color='C1',
            marker='o',
            markersize=5,
            linestyle=None,
            alpha=1)

    ax1.plot(np.arange(len(da_obs_detrend.time))[-1],
            da_obs_detrend[-1],
            color='C0',
            marker='o',
            markersize=5,
            linestyle=None,
            alpha=1)

    # dot for the last element
    ax1.plot(np.arange(lastind,lastind+12)[0],
            ds_nmme['global'].mean(dim='M').mean(dim='model')[0],
            color='C1',
            marker='o',
            markersize=5,
            linestyle=None,
            alpha=1)

    ax1.plot(np.arange(lastind,lastind+12)[0],
            ds_nmme['global_detrend'].mean(dim='M').mean(dim='model')[0],
            color='C0',
            marker='o',
            markersize=5,
            linestyle=None,
            alpha=1)


    # # horizontal line
    # ax1.axhline(y=da_perc_obs_ts[-1], color='C1', linestyle='--',linewidth=2)
    # ax1.axhline(y=da_perc_obs_ts_detrend[-1], color='C0', linestyle='--',linewidth=2)



    # # ranking text outside the plot
    # ax1.text(np.arange(len(da_perc_obs_ts.time))[-1]+5+12,
    #          da_perc_obs_ts[-1]+0.01,
    #          f'{forecasttime[-1]} ',
    #          color='C1',
    #          size=13)
    # ax1.text(np.arange(len(da_perc_obs_ts.time))[-1]+5+12,
    #          da_perc_obs_ts[-1]-0.02,
    #          f'ranks {ranking} of {len(da_perc_obs_ts.time)} months',
    #          color='C1',
    #          size=13)
    # ax1.text(np.arange(len(da_perc_obs_ts_detrend.time))[-1]+5+12,
    #          da_perc_obs_ts_detrend[-1]+0.01,
    #          f'{forecasttime[-1]}',
    #          color='C0',
    #          size=13)
    # ax1.text(np.arange(len(da_perc_obs_ts_detrend.time))[-1]+5+12,
    #          da_perc_obs_ts_detrend[-1]-0.02,
    #          f'ranks {detrend_ranking} of {len(da_perc_obs_ts_detrend.time)} months',
    #          color='C0',
    #          size=13)

    # calculate the ranking of the with trend time series
    forecast_ranking = find_rank(
        np.append(
            da_obs.data,
            ds_nmme['global'].mean(dim='M').mean(dim='model')[0].data
        )
    )

    # calculate the ranking of the without trend time series
    forecast_detrend_ranking = find_rank(
        np.append(
            da_obs_detrend.data,
            ds_nmme['global_detrend'].mean(dim='M').mean(dim='model')[0].data
        )
    )

    # ranking text outside the plot
    ax1.text(np.arange(len(da_obs.time))[-1]+5+12,
            da_obs[-1]+0.01,
            f'{forecasttime[-12]} forecast',
            color='C1',
            size=13)
    ax1.text(np.arange(len(da_obs.time))[-1]+5+12,
            da_obs[-1]-0.02,
            f'ranks {forecast_ranking} of {len(da_obs.time)+1} months',
            color='C1',
            size=13)
    ax1.text(np.arange(len(da_obs_detrend.time))[-1]+5+12,
            da_obs_detrend[-1]+0.01,
            f'{forecasttime[-12]} forecast',
            color='C0',
            size=13)
    ax1.text(np.arange(len(da_obs_detrend.time))[-1]+5+12,
            da_obs_detrend[-1]-0.02,
            f'ranks {forecast_detrend_ranking} of {len(da_obs_detrend.time)+1} months',
            color='C0',
            size=13)


    #### setting the plotting format
    ax1.set_ylabel('Ocean area percentage (%)',{'size':'18'},color='k')
    ax1.set_ylim([0,0.5])
    ax1.set_xlim([0,len(da_obs.time)+12])
    ax1.set_xlabel('Time',{'size':'18'},labelpad=10)
    ax1.tick_params(axis='y',labelsize=20,labelcolor='k')
    ax1.tick_params(axis='x',labelsize=12,labelcolor='k',rotation=70)
    ax1.set_xticks(np.arange(len(da_obs.time)+12)[::12])
    ax1.set_xticklabels(forecastyear[::12])
    ax1.set_yticks(np.arange(0,0.6,0.1))
    ax1.set_yticklabels([f"{int(n*100)}" for n in np.arange(0,0.6,0.1)])
    ax1.set_title(
        "Global Ocean MHW Area",
        color='black',
        weight='bold',
        size=20,
        pad=20
    )
    legend = ax1.legend(loc='upper right',fontsize=14,frameon=False)
    legend.set_bbox_to_anchor((1.36, 0.2))

    return fig, ax1

def plot_history(da_obs,da_obs_detrend,RANK=True,LEGEND=True):
    """
    Function to plot the historical plot only
    """
    fig=plt.figure(1,figsize=(15,12))
    ax1=fig.add_axes([0,0.75,1,0.3])

    forecasttime = []
    forecastyear= []
    for btime in da_obs.time :
        new_date = (
            date.fromisoformat(f'{btime.dt.year.data}-'+
                            f'{btime.dt.month.data:02d}-'+
                            f'{btime.dt.day.data:02d}')
        )
        forecasttime.append(new_date.strftime("%b %Y"))
        forecastyear.append(new_date.strftime("%Y"))

    # time series
    ax1.plot(np.arange(len(da_obs.time)),
            da_obs,
            label='With warming trend',
            linewidth=2,
            color='C1',
            alpha=0.6)

    ax1.plot(np.arange(len(da_obs_detrend.time)),
            da_obs_detrend,
            label='Without warming trend',
            linewidth=2,
            color='C0',
            alpha=0.6)

    # dot for the last element
    ax1.plot(np.arange(len(da_obs.time))[-1],
            da_obs[-1],
            color='C1',
            marker='o',
            markersize=12,
            linestyle=None,
            alpha=1)

    ax1.plot(np.arange(len(da_obs_detrend.time))[-1],
            da_obs_detrend[-1],
            color='C0',
            marker='o',
            markersize=12,
            linestyle=None,
            alpha=1)

    # horizontal line
    ax1.axhline(y=da_obs[-1], color='C1', linestyle='--',linewidth=2)
    ax1.axhline(y=da_obs_detrend[-1], color='C0', linestyle='--',linewidth=2)

    # calculate the ranking of the with trend time series
    ranking = find_rank(da_obs.data)

    # calculate the ranking of the without trend time series
    detrend_ranking = find_rank(da_obs_detrend.data)

    if RANK:
        # ranking text outside the plot
        ax1.text(np.arange(len(da_obs.time))[-1]+5,
                da_obs[-1]+0.01,
                f'{forecasttime[-1]} ',
                color='C1',
                size=13)
        ax1.text(np.arange(len(da_obs.time))[-1]+5,
                da_obs[-1]-0.02,
                f'ranks {ranking} of {len(da_obs.time)} months',
                color='C1',
                size=13)
        ax1.text(np.arange(len(da_obs_detrend.time))[-1]+5,
                da_obs_detrend[-1]+0.01,
                f'{forecasttime[-1]}',
                color='C0',
                size=13)
        ax1.text(np.arange(len(da_obs_detrend.time))[-1]+5,
                da_obs_detrend[-1]-0.02,
                f'ranks {detrend_ranking} of {len(da_obs_detrend.time)} months',
                color='C0',
                size=13)

    #### setting the plotting format
    ax1.set_ylabel('Ocean area percentage (%)',{'size':'18'},color='k')
    ax1.set_ylim([0,0.5])
    ax1.set_xlim([0,len(da_obs.time)+2])
    ax1.set_xlabel('Time',{'size':'18'},labelpad=10)
    ax1.tick_params(axis='y',labelsize=20,labelcolor='k')
    ax1.tick_params(axis='x',labelsize=12,labelcolor='k',rotation=70)
    ax1.set_xticks(np.arange(len(da_obs.time))[::12])
    ax1.set_xticklabels(forecastyear[::12])
    ax1.set_yticks(np.arange(0,0.6,0.1))
    ax1.set_yticklabels([f"{int(n*100)}" for n in np.arange(0,0.6,0.1)])
    ax1.set_title(
        "Global Ocean MHW area (OISSTv2)",
        color='black',
        weight='bold',
        size=20,
        pad=20
    )
    if LEGEND:
        legend = ax1.legend(loc='upper right',fontsize=14,frameon=False)
        legend.set_bbox_to_anchor((1.36, 0.2))

    return fig, ax1

def plot_noaa_em(fig,set_ax=None):
    """
    Plotting the NOAA emblem on the plot
    """
    if set_ax is None:
        set_ax=[0,0,1,1]
    ax = fig.add_axes(set_ax)
    im = image.imread('/Datasets.private/marinehw/noaa_web.png')
    ax.imshow(im, zorder=-1)
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig

def save_obs_data(
        da_mhw_perc : xr.DataArray,
        da_mhw_detrend_perc : xr.DataArray,
        output_dir : str = "/home/chsu/MHW/"):
    """
    Function for saving the observational data
    which is used to derived the MHW area coverage 

    Parameters
    ----------
    output_dir : str
    The output directory. Default to "/home/chsu/MHW/"
    
    Returns
    -------

    Raises
    ------
    """
    ds_obs_ts = xr.Dataset()
    ds_obs_ts['oisst_mhw_areaperc'] = da_mhw_perc
    ds_obs_ts['oisst_mhw_areaperc_detrended'] = da_mhw_detrend_perc

    FILENAME = 'oisst_mhw_areaperc.nc'

    # dataset attrs
    ds_obs_ts.attrs['filename'] = FILENAME
    ds_obs_ts.attrs['history'] = "derived at PSL from OISSTv2"
    ds_obs_ts.attrs['comment'] = "figure available at https://psl.noaa.gov/marine-heatwaves/#report"
    ds_obs_ts.attrs['contact'] = "chia-wei.hsu@noaa.gov"

    # variable attrs
    ds_obs_ts['oisst_mhw_areaperc'].attrs['long_name'] = "area percentage of marine heatwave over the global ocean based on OISST anomaly"
    ds_obs_ts['oisst_mhw_areaperc'].attrs['var_desc'] = "area percentage of marine heatwave over the global ocean derived from OISSTv2"
    ds_obs_ts['oisst_mhw_areaperc'].attrs['dataset'] = "OISSTv2 observed Marine Heatwave"
    ds_obs_ts['oisst_mhw_areaperc'].attrs['statistic'] = "the marine heatwave is defined based on 90% threshold from OISST monthly mean with a base climatology from 1991-2020"
    ds_obs_ts['oisst_mhw_areaperc'].attrs['parent_stat'] = "Observed Values"
    ds_obs_ts['oisst_mhw_areaperc'].attrs['units'] = "%(unitless)"

    ds_obs_ts['oisst_mhw_areaperc_detrended'].attrs['long_name'] = "area percentage of marine heatwave over the global ocean based on detrended OISST anomaly"
    ds_obs_ts['oisst_mhw_areaperc_detrended'].attrs['var_desc'] = "area percentage of marine heatwave over the global ocean derived from OISSTv2"
    ds_obs_ts['oisst_mhw_areaperc_detrended'].attrs['dataset'] = "OISSTv2 observed Marine Heatwave"
    ds_obs_ts['oisst_mhw_areaperc_detrended'].attrs['statistic'] = "the marine heatwave is defined based on 90% threshold from OISST monthly mean with a base climatology from 1991-2020"
    ds_obs_ts['oisst_mhw_areaperc_detrended'].attrs['parent_stat'] = "Observed Values"
    ds_obs_ts['oisst_mhw_areaperc_detrended'].attrs['units'] = "%(unitless)"

    # time attrs
    ds_obs_ts['time'].attrs['long_name'] = "Time"
    # ds_obs_ts['time'].attrs['units'] = "days since 1800-01-01 00:00:00"
    ds_obs_ts['time'].attrs['delta_t'] = "0000-01-00 00:00:00"
    ds_obs_ts['time'].attrs['avg_period'] = "0000-01-00 00:00:00"
    ds_obs_ts['time'].attrs['T'] = "T"

    ds_obs_ts = ds_obs_ts.drop_vars(["month"])
    ds_obs_ts.to_netcdf(f'{output_dir}{FILENAME}')

def save_nmme_data(
        da_nmme : xr.DataArray,
        detrend : bool = False,
        output_dir : str = '/home/chsu/MHW/'
    ):
    """
    The function to save the NMME model data (plume plot)

    Parameters
    ----------
    da_nmme : xr.DataArray
        The xr.DataArray object that store the NMME models which 
        includes the individual model and each ensemble member of 
        area percentage of MHW over the global ocean.

    detrend : bool 
        The boolean object that determine if the saved NMME result
        is based on detrended MHW or MHW with the original trend 
        included. Default value is False.

    output_dir : str
        The output directory. Default to "/home/chsu/MHW/"

    Returns
    -------

    Raises
    ------

    """
    ds_nmme_data = xr.Dataset()
    ds_nmme_data['nmme'] = da_nmme

    if detrend:
        FILENAME_ATTR = 'detrend'
    else :
        FILENAME_ATTR = 'trend'

    for _,model in enumerate(da_nmme.model.data):
        # calculate ensemble mean of each model
        da_model_ens = (
            da_nmme
            .sel(model=model)
            .mean(dim='M')
            .compute()
        )
        ds_nmme_data[f'nmme_{model}_mean'] = da_model_ens

        # variable attrs
        ds_nmme_data[f'nmme_{model}_mean'].attrs['long_name'] = f"area percentage of marine heatwave ({FILENAME_ATTR}) over the global ocean based on {model} ensemble mean"
        ds_nmme_data[f'nmme_{model}_mean'].attrs['var_desc'] = f"area percentage of marine heatwave ({FILENAME_ATTR}) over the global ocean derived from {model} ensemble mean"
        ds_nmme_data[f'nmme_{model}_mean'].attrs['dataset'] = f"{model} Simulated Marine Heatwave ({FILENAME_ATTR})"
        ds_nmme_data[f'nmme_{model}_mean'].attrs['statistic'] = f"the marine heatwave ({FILENAME_ATTR}) is defined based on 90% threshold from {model} monthly mean with a base climatology from 1991-2020"
        ds_nmme_data[f'nmme_{model}_mean'].attrs['parent_stat'] = "NMME Simulation Values"
        ds_nmme_data[f'nmme_{model}_mean'].attrs['units'] = "%(unitless)"

    # calculate multi-model mean of ensemble mean
    da_mmm = (
        da_nmme
        .mean(dim='M')
        .mean(dim='model')
        .compute()
    )
    ds_nmme_data['nmme_mmm'] = da_mmm

    # variable attrs
    ds_nmme_data['nmme_mmm'].attrs['long_name'] = f"area percentage of marine heatwave ({FILENAME_ATTR}) over the global ocean based on NMME multi-model ensemble mean"
    ds_nmme_data['nmme_mmm'].attrs['var_desc'] = f"area percentage of marine heatwave ({FILENAME_ATTR}) over the global ocean derived from NMME multi-model ensemble mean"
    ds_nmme_data['nmme_mmm'].attrs['dataset'] = f"NMME multi-model ensemble mean Simulated Marine Heatwave ({FILENAME_ATTR})"
    ds_nmme_data['nmme_mmm'].attrs['statistic'] = f"the marine heatwave ({FILENAME_ATTR}) is defined based on 90% threshold from NMME multi-model ensemble mean monthly mean with a base climatology from 1991-2020"
    ds_nmme_data['nmme_mmm'].attrs['parent_stat'] = "NMME Simulation Values"
    ds_nmme_data['nmme_mmm'].attrs['units'] = "%(unitless)"

    FILENAME = f'nmme_mhw_areaperc_{FILENAME_ATTR}.nc'

    # dataset attrs
    ds_nmme_data.attrs['filename'] = FILENAME
    ds_nmme_data.attrs['history'] = "derived at PSL from NMME"
    ds_nmme_data.attrs['comment'] = "figure available at https://psl.noaa.gov/marine-heatwaves/#report"
    ds_nmme_data.attrs['contact'] = "chia-wei.hsu@noaa.gov"

    # time attrs
    ds_nmme_data['S'].attrs['long_name'] = "Initial time"
    ds_nmme_data.to_netcdf(f'{output_dir}{FILENAME}')


##################################     Main script     ######################################
if __name__ == "__main__":
    ### open local cluster
    client = Client(processes=False)
    OISST_OBS_DATA = True
    NMME_MODEL_DATA = True
    OUTPUT_DATA_DIR = '/Public/chsu/share_mhw/'
    PREMONTH = 24                          # historical record to include in plume plot (month)
    # PROC_TIME = '24_01_22'               # new OISST file that include start from 1982
    # OUTPUTDIR = '/home/chsu/MHW/figure/'
    PROC_TIME = 'latest'
    OUTPUTDIR = '/httpd-test/psd/marine-heatwaves/img/'

    #### Calculating area percentage for observational data
    obs_mask_dict = all_obs_area_mask(proc_time=PROC_TIME)
    da_mask_01 = obs_mask_dict['mhw_trend']
    da_mask_01_detrend = obs_mask_dict['mhw_detrended']
    da_unmask_01 = obs_mask_dict['global']

    # mhw area
    da_mask_ts = area_weighted_sum(da_mask_01, xname='lon', yname='lat')
    da_mask_ts_detrend = area_weighted_sum(da_mask_01_detrend, xname='lon', yname='lat')

    # global ocean area
    da_unmask_ts = area_weighted_sum(da_unmask_01, xname='lon', yname='lat')
    da_unmask_ts = da_unmask_ts.drop_vars('month')

    # mhw area percentage
    da_perc_obs_ts = da_mask_ts/da_unmask_ts
    da_perc_obs_ts_detrend = da_mask_ts_detrend/da_unmask_ts

    if OISST_OBS_DATA:
        save_obs_data(da_perc_obs_ts,da_perc_obs_ts_detrend,output_dir=OUTPUT_DATA_DIR)

    #### Calculating area percentage for NMME with trend/detrended
    nmme_mask_dict = nmme_newest_forecast(threshold=90)
    da_mask_01 = nmme_mask_dict['mhw_trend']
    da_unmask_01 = nmme_mask_dict['global']

    # mhw area
    da_mask_ts = area_weighted_sum(da_mask_01, xname='X', yname='Y')
    da_mask_ts = da_mask_ts*nmme_mask_dict['ens_mask']

    # global ocean area
    da_unmask_ts = area_weighted_sum(da_unmask_01, xname='X', yname='Y')

    # store area percentage
    ds_perc = xr.Dataset()
    ds_perc['global'] = (da_mask_ts/da_unmask_ts).compute()

    # mhw area detrended
    da_mask_01 = nmme_mask_dict['mhw_detrended']
    da_mask_ts = area_weighted_sum(da_mask_01, xname='X', yname='Y')
    da_mask_ts = da_mask_ts*nmme_mask_dict['ens_mask']
    ds_perc['global_detrend'] = (da_mask_ts/da_unmask_ts).compute()

    # create valid-time (leadtime) time stamp
    ini_year = ds_perc.S.dt.year.data
    ini_month = ds_perc.S.dt.month.data
    valid_time = xr.cftime_range(start=f'{ini_year}-{ini_month:02d}-01', periods=13, freq='MS')[:-1]
    ds_perc['valid_time'] = xr.DataArray(valid_time,coords={'L':ds_perc['L']},dims={'L'})
    ds_perc = ds_perc.set_coords('valid_time')

    #### plot with trend plume plot
    print('PLOT WITH TREND')
    fig2, ax2 = plot_plume(ds_perc['global'],da_perc_obs_ts,PREVMONTH=PREMONTH)
    ax2.set_title("Global Ocean MHW Area", color='black', weight='bold',size=15,pad=20)
    fig2 = plot_noaa_em(fig2,set_ax=[1,0.75-0.10,0.1,0.1])
    fig2.savefig(f'{OUTPUTDIR}/MHW_area_plume.png',
            dpi=300,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            transparent=False,
            bbox_inches="tight",
            pad_inches=None)
    plt.clf()

    print('PLOT WITHOUT TREND')
    fig3, ax3 = plot_plume(ds_perc['global_detrend'],da_perc_obs_ts_detrend,PREVMONTH=PREMONTH)
    ax3.set_title(
        "Global Ocean MHW Area (without trend)",
        color='black',
        weight='bold',
        size=15,
        pad=20
    )
    fig3 = plot_noaa_em(fig3,set_ax=[1,0.75-0.1,0.1,0.1])
    fig3.savefig(f'{OUTPUTDIR}/MHW_area_plume_detrend.png',
            dpi=300,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            transparent=False,
            bbox_inches="tight",
            pad_inches=None)
    plt.clf()

    # fig4, ax4 = plot_history_forecast(ds_perc,da_perc_obs_ts,da_perc_obs_ts_detrend)
    # fig4.savefig('/home/chsu/MHW/figure/MHW_area_history_forecast.png',
    #         dpi=300,
    #         facecolor='w',
    #         edgecolor='w',
    #         orientation='portrait',
    #         transparent=False,
    #         bbox_inches="tight",
    #         pad_inches=None)
    # plt.clf()

    fig5, ax5 = plot_history(da_perc_obs_ts,da_perc_obs_ts_detrend)
    fig5 = plot_noaa_em(fig5,set_ax=[1.25,0.75-0.1,0.1,0.1])
    fig5.savefig(f'{OUTPUTDIR}/MHW_area_history_only.png',
            dpi=300,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            transparent=False,
            bbox_inches="tight",
            pad_inches=None)
    plt.clf()

    if NMME_MODEL_DATA:
        save_nmme_data(ds_perc['global'],detrend=False,output_dir=OUTPUT_DATA_DIR)
        save_nmme_data(ds_perc['global_detrend'],detrend=True,output_dir=OUTPUT_DATA_DIR)