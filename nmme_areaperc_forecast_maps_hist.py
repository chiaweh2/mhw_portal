"""
The script is designed to produced MHW area percentage based on NMME.

"""

from datetime import date,datetime
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from dask.distributed import Client
from dateutil.relativedelta import relativedelta
from nmme_areaperc_climo import woa09_ocean_mask
from nmme_areaperc_history_oisst_forecast_plume import plot_noaa_em
from nmme_download import iri_nmme_models
from nmme_areaperc_forecast_maps_hist_uswest import read_nmme_onlist_forecast

import warnings
warnings.filterwarnings("ignore")

def plot_glo_map(
    ax,
    da,
    clevels=range(0,10),
    res='110m',
    ytick=True,
    ytickleft=True,
    cmap='inferno_r'
):
    '''
    This is a function plot the map of the MHW report
    2x4 maps and individual map detail is set in this 
    function.

    Parameters
    ----------
    ax : matplotlib axes object 
        The axes is created before the specific is set in the function
    da : xr.DataArray
        The data array is used to be plot on the map with pcolormesh
    clevels : arrays (python list or numpy array)
		The level used in the colorbar which determine the plotted 
        values. 
    res : string
		The string used to indicate the resolution of the coastline and
        continent. options are '10m', '50m', or '110m', default is '110m'.
    ytick : boolean
        The boolean is to set if the ytick need to be on the map or not
    ytickleft : boolean
        The boolean only works when ytick is set to True. This kwarg determine 
        if the ytick will be on the left side of the map or not. Set to True if 
        one want the tick to be on the left. Set to False if the ytick need to 
        be on the left of the map. Default value is True.
    cmap : string
        The string for the colormap options in matplotlib
        https://matplotlib.org/stable/tutorials/colors/colormaps.html


    Returns
    -------
    ax : matplotlib axes object
        The detail setting is applied on the axes.
    im : matplotlib image object on the axes
        The image (map) output is for user to set the desired colorbar
        correpsonds to the image.

    '''
    im = da.plot.pcolormesh(
        ax=ax,
        levels=clevels,
        extend='both',
        cmap=cmap,
        transform=ccrs.PlateCarree(central_longitude=0.)
    )

    cb=im.colorbar
    cb.remove()
    ax.coastlines(resolution=res,linewidths=0.5)

    land_50m = cfeature.NaturalEarthFeature('physical', 'land', res,
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'])

    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale=res,
            facecolor='none')

    ax.add_feature(land_50m,color='lightgrey',linewidth=0.5)
    ax.add_feature(states_provinces,edgecolor='grey',linewidth=0.2)
    ax.add_feature(cfeature.BORDERS,linewidth=0.2)

    ax.set_xticks([60,120,180,240,300], crs=ccrs.PlateCarree())
    ax.set_xticklabels([60,120,180,-120,-60], color='black', weight='bold',size=22)
    if ytick:
        ax.set_yticks([-80,-60,-40,-20,0,20,40,60,80], crs=ccrs.PlateCarree())
        ax.set_yticklabels([-80,-60,-40,-20,0,20,40,60,80], color='black', weight='bold',size=22)
    else:
        ax.set_yticks([], crs=ccrs.PlateCarree())
        ax.set_yticklabels([], color='black', weight='bold',size=22)

    if ytickleft:
        ax.yaxis.tick_left()
    else:
        ax.yaxis.tick_right()

    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xlabel('')
    ax.set_ylabel('')

    return ax,im

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

    ### open local cluster
    client = Client(processes=False)

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
    all_avai_model_list = list(dict_model.keys())

    da_prob,da_anomoly,da_mhw_anom,da_num_mem,da_ensmask = read_nmme_onlist_forecast(
        cyear,
        cmonth,
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
            ax2.set_title(f'Probability (NMME start {forecasttime[0]})',
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
            ax2.set_title('Magnitude',
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
    # fig.savefig('/httpd-test/psd/marine-heatwaves/img/MHW_maps.png', dpi=150, facecolor='w', edgecolor='w',
    #                 orientation='portrait', format=None,
    #                 transparent=False, bbox_inches="tight", pad_inches=None)
    fig.savefig('/home/chsu/mhw_portal/figures/MHW_maps.png', dpi=150, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches="tight", pad_inches=None)


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
        climo_areaperc_file = f'{PREDIR}NMME_MHW_maskEneMean_areaPerc_climo_1991_2020_'
        for nmodel, model in enumerate(model_use_list):
            if model != model_use_list[-1]:
                climo_areaperc_file = climo_areaperc_file+model+'_'
            else:
                climo_areaperc_file = climo_areaperc_file+model+'.nc'

        ds_perc_climo = xr.open_dataset(climo_areaperc_file)
        ds_perc_climo = ds_perc_climo.where(ds_perc_climo!=0)
        # pick the same month in climo file
        month = da_mask_ts.month.data
        da_perc_climo = ds_perc_climo.sel(month=month)
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

    fig.savefig(
        '/home/chsu/mhw_portal/figures/MHW_area_bar.png',
        dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait',
        transparent=False,
        bbox_inches="tight",
        pad_inches=None
    )
    # fig = plot_noaa_em(fig,set_ax=[1.2,0.75-0.11,0.1,0.1])
    # fig.savefig('/httpd-test/psd/marine-heatwaves/img/MHW_area_bar.png', dpi=300, facecolor='w', edgecolor='w',
    #                 orientation='portrait',
    #                 transparent=False, bbox_inches="tight", pad_inches=None)
