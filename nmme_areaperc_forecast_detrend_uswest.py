"""
The script is designed to produced MHW area percentage based on NMME.

"""
import json
import warnings
from datetime import date,datetime,timedelta
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from dask.distributed import Client
from dateutil.relativedelta import relativedelta
from nmme_download import iri_nmme_models
from nmme_areaperc_climo import read_eez, region_mask
from nmme_areaperc_history_oisst_forecast_plume import area_weighted_sum
from nmme_areaperc_forecast_uswest import plot_region_map
from nmme_areaperc_forecast_detrend import read_nmme_onlist_forecast_detrend

warnings.filterwarnings("ignore")



################################## Main program start #####################################
if __name__ == '__main__':

    ###### Setting ######
    # whether to include the climatology bar in the bar plot
    CLIMO = True

    # regional map limit
    region_limit = [-160,-115,25,60]

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

    x_list_mj,y_list_mj = read_eez()

    da_region = region_mask(x_list_mj,y_list_mj,da_prob,xname='X',yname='Y')
    # if da_mhw_anom['X'].min().data<0:
    #     # change mask coordinate to -180-180
    #     da_region['X'] = da_region['X'].where(
    #         da_region['X']<=180.,
    #         other=da_region['X']-360.
    #     )


    #### Calculating area percentage
    da_global = da_num_mem.where(da_num_mem==0,other=1.)
    da_mhw_mask_all = da_mhw_anom.where(da_mhw_anom.isnull(),other=1)

    # use latest probability and ensemble member to calculate area percentage
    da_mask_01 = da_mhw_mask_all*da_region
    da_unmask_01 = da_global*da_region

    # calculate weight
    weights = np.cos(np.deg2rad(da_unmask_01.Y))
    weights.name = "weights"

    ds_perc = xr.Dataset()

    da_mask_ts = area_weighted_sum(da_mask_01, xname='X', yname='Y')
    da_mask_ts = da_mask_ts*da_ensmask
    
    da_unmask_ts = area_weighted_sum(da_unmask_01, xname='X', yname='Y')

    da_perc_ts = da_mask_ts/da_unmask_ts

    da_perc_mean_ts = da_perc_ts.mean(dim='M')
    da_perc_mean_ts = da_perc_mean_ts.mean(dim='model')
    ds_perc['global'] = da_perc_mean_ts.compute()


    # read in climo area data
    if CLIMO:
        ds_perc_climo = xr.open_dataset(
            '/Datasets.private/marinehw/NMME_preprocess/'+
            'NMME_MHW_maskEneMean_areaPercEEZ_climo_detrend_1991_2020_newgenCanModel_plusCESM1_plusCCSM4.nc'
        )
        ds_perc_climo = ds_perc_climo.where(ds_perc_climo!=0)
        # pick the same month in climo file
        month = da_mask_ts.month.data
        da_perc_climo = ds_perc_climo.sel(month=month)
        da_perc_climo = da_perc_climo.mean(dim='M')
        da_perc_climo = da_perc_climo.mean(dim='model').compute()


    #### Report maps
    fig = plt.figure(2,figsize=(15,10))
    mlevel = np.arange(0,1.6,0.1)
    plevel = np.arange(0,1,0.1)
    GAP = 0.55
    HGAP = 0.45

    leadtime = [9,6,3,1]
    btime = da_prob.S.dt
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
        ax2 = fig.add_axes([0,0+GAP*nlead,1,0.5],projection=ccrs.PlateCarree(central_longitude=(region_limit[0]+region_limit[1])/2))

        da2 = (da_prob).isel(L=lead).sortby('X')
        if lead == leadtime[0] :
            ax2,im2 = plot_region_map(ax2,da2,clevels=plevel,ytick=False,cmap='inferno')
        else:
            ax2,im2 = plot_region_map(ax2,da2,clevels=plevel,xtick=False,ytick=False,cmap='inferno')
            
        ax2.plot(x_list_mj,y_list_mj,color='lime', linestyle='dashed',markersize=0.0,linewidth=2.1,transform=ccrs.PlateCarree())
        ax2.set_extent(region_limit,
                    crs=ccrs.PlateCarree())
        ax2.text(
            -30,
            30,
            f'{forecasttime[lead]} Forecast',
            rotation=90,
            fontsize=25,
            weight='bold'
        )

        # title
        if lead == leadtime[-1] :
            ax2.set_title(f'Probability (start {forecasttime[0]})',
                        pad=20,
                        weight='bold',
                        size=20)
        else:
            ax2.set_title('')

        # colorbar
        if lead == leadtime[0] :
            cbaxes=fig.add_axes([0.38,-0.15,0.4,0.02])
            cbar=fig.colorbar(im2,cax=cbaxes,orientation='horizontal')
            cbar.set_ticks(plevel)
            cbar.set_ticklabels([f"{int(n*100):d}" for n in plevel])
            cbar.ax.tick_params(labelsize=25,rotation=0)
            cbar.set_label(label='MHW probability (%)',size=20, labelpad=15)

    for nlead,lead in enumerate(leadtime):
        ax2 = fig.add_axes([HGAP,0+GAP*nlead,1,0.5],
                        projection=ccrs.PlateCarree(central_longitude=(region_limit[0]+region_limit[1])/2))
        da2 = (da_anomoly).isel(L=lead).mean(dim='M').mean(dim='model').sortby('X')
        if lead == leadtime[0] :
            ax2,im2 = plot_region_map(ax2,da2,clevels=mlevel,ytickleft=False,cmap='OrRd')
        else:
            ax2,im2 = plot_region_map(ax2,da2,clevels=mlevel,xtick=False,ytickleft=False,cmap='OrRd')
        ax2.plot(x_list_mj,y_list_mj,color='lime', linestyle='dashed',markersize=0.0,linewidth=2.1,transform=ccrs.PlateCarree())
        ax2.set_extent(region_limit,
                    crs=ccrs.PlateCarree())
        if lead == leadtime[-1] :
            ax2.set_title('Magnitude',
                        pad=20, weight='bold',size=20)
        else:
            ax2.set_title('')

        # colorbar
        if lead == leadtime[0] :
            cbaxes=fig.add_axes([0.38+HGAP,-0.15,0.4,0.02])
            cbar=fig.colorbar(im2,cax=cbaxes,orientation='horizontal')
            cbar.set_ticks(mlevel)
            cbar.set_ticklabels([f"{n:.1f}" for n in mlevel])
            cbar.ax.tick_params(labelsize=20,rotation=60)
            cbar.set_label(label='MHW magnitude (degC)',size=20, labelpad=15)

    fig.savefig('/home/chsu/mhw_portal/figures/MHW_maps_detrend_eez.png', dpi=150, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches="tight", pad_inches=None)


    #### Report bars
    fig=plt.figure(1,figsize=(9,12))
    ax1=fig.add_axes([0,0.75,1,0.3])

    # World
    ax1.bar(ds_perc['global'].L,
            ds_perc['global'],
            label='EEZ',
            color='k',
            width=0.8,
            alpha=0.5)
    if CLIMO:
        ax1.bar(da_perc_climo['eez'].L,
                da_perc_climo['eez'],
                label='EEZ climatology ',
                color='k',
                width=0.5,
                alpha=1)

    # setting the plotting format
    ax1.set_ylabel('EEZ area percentage (%)',{'size':'18'},color='k')
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

    fig.savefig(
        '/home/chsu/mhw_portal/figures/MHW_area_bar_detrend_eez.png',
        dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait',
        transparent=False,
        bbox_inches="tight",
        pad_inches=None
    )

    client.close()
