"""
The script is designed to produced MHW area percentage based on 
detrended NMME.

"""
import json
import warnings
from datetime import date,datetime,timedelta
import cftime
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from dask.distributed import Client
from dateutil.relativedelta import relativedelta
from nmme_areaperc_forecast import woa09_ocean_mask, plot_glo_map
from nmme_areaperc_history_oisst_forecast_plume import plot_noaa_em
from nmme_climo import read_nmme, iri_nmme_models

warnings.filterwarnings("ignore")
##################################  function  #####################################
def read_nmme_onlist_forecast_detrend(
        current_year: int,
        current_month: int,
        previous_year: int,
        previous_month: int,
        model_list: list[str],
        all_model_list: list[str],
        basedir: str,
        predir: str,
        threshold: int = 90
) -> dict:
    """read in the NMME for MHW area percentage count

    Parameters
    ----------
    current_year: int
        current year that helps extract from the 
        latest model forecast
    current_month: int
        current month that helps extract from the 
        latest model forecast
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
    threshold : int
        the threshold of determine the MHW

    Returns
    -------
    da_mhw_prob :
        a xr.DataArray represents the MHW probability
    da_anom_all :
        a xr.DataArray represents the sst anomaly for all model 
        in the latest update
    da_mhw_anom_all :
        a xr.DataArray represents the sst anomaly only over the 
        identified mhw region for all model in the latest update
    da_nmem_all_out :
        a xr.DataArray represents the number of ensemble members
        in all model in the latest update
    da_ensmask_all : 
        a xr.DataArray represents the mask based on the number of 
        available models in the latest update

    """

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

            try:
                da_model = da_model.sel(
                    S=slice(
                        cftime.Datetime360Day(previous_year, previous_month, 1, 0, 0, 0, 0, has_year_zero=True),
                        cftime.Datetime360Day(current_year, current_month, 1, 0, 0, 0, 0, has_year_zero=True)
                    )
                )
            except KeyError:
            # if any of the date in slice does not exist, the data is fill with NaN
                da_model = da_model.isel(S=range(-2,0))*np.nan
                da_model['S'] = [
                    cftime.Datetime360Day(previous_year, previous_month, 1, 0, 0, 0, 0, has_year_zero=True),
                    cftime.Datetime360Day(current_year, current_month, 1, 0, 0, 0, 0, has_year_zero=True)
                ]
            if len(da_model.S) != 2:
                 # if only one date in slice does exist, the data is still fill with NaN
                da_model = ds_nmme['sst'].isel(S=range(-2,0))*np.nan
                da_model['S'] = [
                    cftime.Datetime360Day(previous_year, previous_month, 1, 0, 0, 0, 0, has_year_zero=True),
                    cftime.Datetime360Day(current_year, current_month, 1, 0, 0, 0, 0, has_year_zero=True)
                ]
                
            # read climatology (1991-2020)
            da_ensmean_climo = xr.open_dataset(climo_file,chunks={'S':1,'L':1})['sst']

            # calculate ensemble member in each model
            da_nmem = da_model.where(da_model.isnull(), other=1).sum(dim=['M']).isel(S=-1)
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

    da_ensmask_all = da_ensmask_all.isel(S=-1).sum(dim=['X','Y']).compute()
    da_ensmask_all = da_ensmask_all.where(da_ensmask_all==0,other=1)
    da_ensmask_all = da_ensmask_all.where(da_ensmask_all==1)

    # create mask for every S, L, X, Y (if model number less than 2 will be masked)
    da_nmodel = (da_nmem_all/da_nmem_all).sum(dim='model')
    da_nmodel_mask = da_nmodel.where(da_nmodel>1)
    da_allmodel_mask = da_nmodel_mask.where(da_nmodel_mask.isnull(),other=1).compute()

    # calculate total member of all model
    da_nmem_all_out = (da_nmem_all*da_allmodel_mask).sum(dim='model').compute()

    # loop through all set threshold
    m = threshold
    da_mhw_list = []
    da_anom_list = []
    da_mhw_anom_list = []
    avai_model_list = []
    for modelname in model_list:
        if modelname in all_model_list:

            # construct model file
            threshold_file = f'{predir}{modelname}_threshold_detrend{m}.nc'
            polyfit_file = f'{predir}{modelname}_polyfit_p.nc'

            print('------------')
            print(modelname,' MHW detection...')
            print('------------')

            # read threshold (1991-2020)
            da_threshold = xr.open_dataset(
                threshold_file,
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
            da_anom_list.append(da_detrend)
            avai_model_list.append(modelname)

            print('calculating MHW')
            month = da_detrend.month.data
            da_mhw_temp = da_detrend.where(da_detrend>=da_threshold.sel(month=month))
            da_mhw_anom_list.append(da_mhw_temp)
            da_mhw = (da_mhw_temp
                        .where(da_mhw_temp.isnull(),other=1)
                        .sum(dim='M',skipna=True)
            )
            da_mhw_list.append(da_mhw)

    da_anom_list = [da for da in da_anom_list if type(da) != type("string")]
    da_mhw_anom_list = [da for da in da_mhw_anom_list if type(da) != type("string")]
    da_mhw_list = [da for da in da_mhw_list if type(da) != type("string")]

    da_anom_all = xr.concat(da_anom_list,dim='model',join='outer')
    da_anom_all['model'] = avai_model_list

    da_mhw_anom_all = xr.concat(da_mhw_anom_list,dim='model',join='outer')
    da_mhw_anom_all['model'] = avai_model_list

    da_mhw_all = xr.concat(da_mhw_list,dim='model',join='outer')
    da_mhw_all_out = (da_mhw_all*da_allmodel_mask).sum(dim='model').compute()
    da_mhw_prob = (da_mhw_all_out/da_nmem_all_out)*da_allmodel_mask

    return da_mhw_prob,da_anom_all,da_mhw_anom_all,da_nmem_all_out,da_ensmask_all

################################## Main program start #####################################
if __name__ == '__main__':

    ###### Setting ######
    # whether to include the climatology bar in the bar plot
    CLIMO = True

    # directory where new simulation (inputs) and mhw forecast (outputs) is located
    BASEDIR = '/Datasets.private/marinehw/nmme_sst_raw/'

    # directory where sst threshold/climatology/trend (inputs) and mhw hindcast (outputs) is located
    PREDIR = '/Datasets.private/marinehw/nmme_sst_stat/'

    # MHW threshold for prediction
    mhw_threshold = 90

    # Get the current date
    today = datetime.now()

    # Get year, month, and day from the current date
    cyear = today.year
    cmonth = today.month
    prev_month_datetime = today.replace(day=1)
    prev_month_datetime = prev_month_datetime - timedelta(days=1)
    pyear = prev_month_datetime.year
    pmonth = prev_month_datetime.month

    ### open local cluster
    client = Client(processes=False)

    # used model list
    with open('model_use_list.json','r',encoding='utf-8') as f:
        json_dict = json.load(f)
    model_use_list = json_dict['model_use_list']

    dict_model = iri_nmme_models()
    all_avai_model_list = list(dict_model.keys())

    da_prob,da_anomoly,da_mhw_anom,da_num_mem,da_ensmask = read_nmme_onlist_forecast_detrend(
        cyear,
        cmonth,
        pyear,
        pmonth,
        model_use_list,
        all_avai_model_list,
        BASEDIR,
        PREDIR,
        threshold = mhw_threshold
    )

    # # Report Figures
    fig = plt.figure(2,figsize=(15,10))
    mlevel = np.arange(0,1.6,0.1)
    plevel = np.arange(0,1,0.1)
    GAP = 0.65
    HGAP = 0.75

    leadtime = [9,6,3,1]
    btime = da_num_mem.S.dt
    forecasttime = []
    for lead in range(0,12):
        new_date = (
            date.fromisoformat(f'{btime.year.data}-'+
                                f'{btime.month.data:02d}-'+
                                f'{btime.day.data:02d}')
            +relativedelta(months=lead)
        )
        forecasttime.append(new_date.strftime("%b %Y"))

    for nlead,lead in enumerate(leadtime):
        ax2 = fig.add_axes([0,0+GAP*nlead,1,0.5],projection=ccrs.PlateCarree(central_longitude=180))
        da2 = da_prob.isel(L=lead)
        ax2,im2 = plot_glo_map(ax2,da2,clevels=plevel,ytick=False,cmap='inferno')
        # ax2.text(-200, -40, f'Lead {lead+0.5}month', rotation=90, fontsize=25, weight='bold')
        ax2.text(
            -200,
            -50,
            f'{forecasttime[lead]} Forecast',
            rotation=90,
            fontsize=25,
            weight='bold'
        )

        # title
        if lead == leadtime[-1] :
            ax2.set_title(f'Detrended probability (NMME start {forecasttime[0]})',
                          pad=20,
                          weight='bold',
                          size=25)
        else:
            ax2.set_title('')

        # colorbar
        if lead == leadtime[0] :
            cbaxes=fig.add_axes([0+0.17,0-0.12,0.6,0.02])
            cbar=fig.colorbar(im2,cax=cbaxes,orientation='horizontal')
            cbar.set_ticks(plevel)
            cbar.set_ticklabels([f"{int(n*100):d}" for n in plevel])
            cbar.ax.tick_params(labelsize=25,rotation=0)
            cbar.set_label(label='MHW probability (%)',size=20, labelpad=15)

    for nlead,lead in enumerate(leadtime):
        ax2 = fig.add_axes([HGAP,0+GAP*nlead,1,0.5],
                           projection=ccrs.PlateCarree(central_longitude=180))
        da2 = (da_anomoly).isel(L=lead).mean(dim='M').mean(dim='model')
        ax2,im2 = plot_glo_map(ax2,da2,clevels=mlevel,ytickleft=False,cmap='OrRd')
        if lead == leadtime[-1] :
            ax2.set_title('Detrended magnitude',
                          pad=20, weight='bold',size=25)
        else:
            ax2.set_title('')

        # colorbar
        if lead == leadtime[0] :
            cbaxes=fig.add_axes([0+HGAP+0.17,0-0.12,0.6,0.02])
            cbar=fig.colorbar(im2,cax=cbaxes,orientation='horizontal')
            cbar.set_ticks(mlevel)
            cbar.set_ticklabels([f"{n:.1f}" for n in mlevel])
            cbar.ax.tick_params(labelsize=25,rotation=40)
            cbar.set_label(label='MHW magnitude (degC)',size=20, labelpad=15)

    fig = plot_noaa_em(fig,set_ax=[2*HGAP+0.05,0-0.23,0.15,0.15])
    fig.savefig('/httpd-test/psd/marine-heatwaves/img/MHW_maps_detrend.png', dpi=150, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches="tight", pad_inches=None)
    # fig.savefig('/home/chsu/mhw_portal/figures/MHW_maps_detrend.png', dpi=150, facecolor='w', edgecolor='w',
    #                 orientation='portrait', format=None,
    #                 transparent=False, bbox_inches="tight", pad_inches=None)


    # Calculating area percentage
    # calculate global ocean mask based on all models, S, L 
    # (with minimum 2 models criteria included)
    da_global = da_num_mem.where(da_num_mem==0,other=1.)

    # use latest probability and ensemble member to calculate area percentage
    da_mhw_mask_all = da_mhw_anom.where(da_mhw_anom.isnull(),other=1)
    da_mask_01 = da_mhw_mask_all
    da_unmask_01 = da_global

    # calculate weight
    weights = np.cos(np.deg2rad(da_unmask_01.Y))
    weights.name = "weights"

    ds_perc = xr.Dataset()

    da_mask_ts = (da_mask_01).weighted(weights).sum(dim=["X","Y"])
    da_unmask_ts = (da_unmask_01).weighted(weights).sum(dim=["X","Y"])

    da_mask_ts = da_mask_ts*da_ensmask

    da_perc_ts = da_mask_ts/da_unmask_ts

    da_perc_mean_ts = da_perc_ts.mean(dim='M')
    da_perc_mean_ts = da_perc_mean_ts.mean(dim='model')
    ds_perc['global'] = da_perc_mean_ts.compute()

    basin_names=['pacific','atlantic','indian']
    for basin in basin_names:
        # calculate basin mask for NMME output
        da_basin = woa09_ocean_mask(da_unmask_01, basinname = basin, var_x='X', var_y='Y')

        da_mask_ts = (da_mask_01*da_basin).weighted(weights).sum(dim=["X","Y"])
        da_unmask_ts = (da_unmask_01*da_basin).weighted(weights).sum(dim=["X","Y"])

        da_mask_ts = da_mask_ts*da_ensmask

        da_perc_ts = da_mask_ts/da_unmask_ts

        da_perc_mean_ts = da_perc_ts.mean(dim='M')
        da_perc_mean_ts = da_perc_mean_ts.mean(dim='model')
        ds_perc[basin] = da_perc_mean_ts.compute()


    # read in climo area data
    if CLIMO:
        climo_areaperc_file = f'{PREDIR}NMME_MHW_maskEneMean_areaPerc_climo_detrend_1991_2020_'
        for nmodel, model in enumerate(model_use_list):
            if model != model_use_list[-1]:
                climo_areaperc_file = climo_areaperc_file+model+'_'
            else:
                climo_areaperc_file = climo_areaperc_file+model+'.nc'

        ds_perc_climo = xr.open_dataset(climo_areaperc_file)
        ds_perc_climo = ds_perc_climo.where(ds_perc_climo!=0)
        # pick the same month in climo file
        mhw_month = da_mask_ts.month.data
        da_perc_climo = ds_perc_climo.sel(month=mhw_month)
        da_perc_climo = da_perc_climo.mean(dim='M')
        da_perc_climo = da_perc_climo.mean(dim='model').compute()

    #### plotting
    fig=plt.figure(1,figsize=(9,12))
    ax1=fig.add_axes([0,0.75,1,0.3])

    ######## Pacific ########
    ax1.bar(ds_perc['pacific'].L-0.3,
            ds_perc['pacific'],
            label='Pacific Ocean ',
            color='C0',
            width=0.2,
            alpha=0.5)
    if CLIMO:
        ax1.bar(da_perc_climo['pacific'].L-0.3,
                da_perc_climo['pacific'],
                label='Pacific Climatology ',
                color='C0',
                width=0.1,
                alpha=1)

    ######## Atlantic ########
    ax1.bar(ds_perc['atlantic'].L-0.1,
            ds_perc['atlantic'],
            label='Atlantic Ocean ',
            color='C1',
            width=0.2,
            alpha=0.5)
    if CLIMO:
        ax1.bar(da_perc_climo['atlantic'].L-0.1,
                da_perc_climo['atlantic'],
                label='Atlantic Climatology ',
                color='C1',
                width=0.1,
                alpha=1)

    ######## Indian ########
    ax1.bar(ds_perc['indian'].L+0.1,
            ds_perc['indian'],
            label='Indian Ocean ',
            color='C4',
            width=0.2,
            alpha=0.5)
    if CLIMO:
        ax1.bar(da_perc_climo['indian'].L+0.1,
                da_perc_climo['indian'],
                label='Indian Climatology ',
                color='C4',
                width=0.1,
                alpha=1)

    ######## World ########
    ax1.bar(ds_perc['global'].L+0.3,
            ds_perc['global'],
            label='Global Ocean ',
            color='k',
            width=0.2,
            alpha=0.5)
    if CLIMO:
        ax1.bar(da_perc_climo['global'].L+0.3,
                da_perc_climo['global'],
                label='Global Climatology ',
                color='k',
                width=0.1,
                alpha=1)

    #### setting the plotting format
    ax1.set_ylabel('Ocean area percentage (%)',{'size':'18'},color='k')
    ax1.set_ylim([0,1])
    ax1.set_xlabel('Forecast time',{'size':'18'},labelpad=10)
    ax1.tick_params(axis='y',labelsize=20,labelcolor='k')
    ax1.tick_params(axis='x',labelsize=12,labelcolor='k',rotation=70)
    ax1.set_xticks(np.arange(0.5,12.5))
    ax1.set_xticklabels(forecasttime)
    ax1.set_yticks(np.arange(0,1.1,0.1))
    ax1.set_yticklabels([f"{int(n*100)}" for n in np.arange(0,1.1,0.1)])
    ax1.set_title(
        f"MHW area (NMME start {forecasttime[0]})",
        color='black', weight='bold', size=20, pad=20
    )
    legend = ax1.legend(loc='upper right',fontsize=14,frameon=False)
    legend.set_bbox_to_anchor((1.32, 0.65))
    ax1.grid(linestyle='dashed')

    # fig.savefig(
    #     '/home/chsu/mhw_portal/figures/MHW_area_bar_detrend.png',
    #     dpi=300, facecolor='w', edgecolor='w',
    #     orientation='portrait',
    #     transparent=False,
    #     bbox_inches="tight",
    #     pad_inches=None
    # )
    fig = plot_noaa_em(fig,set_ax=[1.2,0.75-0.11,0.1,0.1])
    fig.savefig('/httpd-test/psd/marine-heatwaves/img/MHW_area_bar_detrend.png', dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait',
                    transparent=False, bbox_inches="tight", pad_inches=None)
    
    client.close()
