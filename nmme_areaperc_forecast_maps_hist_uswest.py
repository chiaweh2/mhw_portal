"""
The script is designed to produced MHW area percentage based on NMME.

"""
import warnings
from datetime import date,datetime
import cftime
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from dask.distributed import Client
from dateutil.relativedelta import relativedelta
from nmme_download import iri_nmme_models
from nmme_climo import read_nmme
from nmme_areaperc_climo import read_eez, region_mask
from nmme_areaperc_history_oisst_forecast_plume import area_weighted_sum

warnings.filterwarnings("ignore")

##################################  function  #####################################
def read_nmme_onlist_forecast(
        current_year: int,
        current_month: int,
        model_list: list[str],
        all_model_list: list[str],
        basedir: str,
        predir: str,
        threshold: int = 90
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
    threshold : int
        the threshold of determine the MHW

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
                    S=cftime.Datetime360Day(
                        current_year, current_month, 1, 0, 0, 0, 0, has_year_zero=True
                    )
                )
            except KeyError:
                # if any of the date in slice does not exist, the data is fill with NaN
                da_model = da_model.isel(S=-1)*np.nan
                da_model['S'] = cftime.Datetime360Day(
                    current_year, current_month, 1, 0, 0, 0, 0, has_year_zero=True
                )

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
            da_anom_list.append(da_anom)
            avai_model_list.append(modelname)

            print('calculating MHW')
            month = da_anom.month.data
            da_mhw_temp = da_anom.where(da_anom>=da_threshold.sel(month=month))
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

def plot_region_map(
    ax,
    da,
    clevels=range(0,10),
    res='110m',
    xtick=True,
    ytick=True,
    ytickleft=True,
    cmap='inferno_r'):

    """
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

    Raises
    ------

    """

    im = da.plot.pcolormesh(
        ax=ax,
        levels=clevels,
        extend='both',
        cmap=cmap,
        transform=ccrs.PlateCarree(central_longitude=0.)
    )

    cb = im.colorbar
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

    if xtick:
        ax.set_xticks([200,210,220,230,240,250], crs=ccrs.PlateCarree())
        ax.set_xticklabels(
            [-160,-150,-140,-130,-120,-110],
            color='black', weight='bold',size=20,rotation=30)
    else:
        ax.set_xticks([], crs=ccrs.PlateCarree())
        ax.set_xticklabels([], color='black', weight='bold',size=20,rotation=30)      

    if ytick:
        ax.set_yticks([20,30,40,50,60], crs=ccrs.PlateCarree())
        ax.set_yticklabels([20,30,40,50,60], color='black', weight='bold',size=20)
    else:
        ax.set_yticks([], crs=ccrs.PlateCarree())
        ax.set_yticklabels([], color='black', weight='bold',size=20)

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

    x_list_mj,y_list_mj = read_eez()

    da_region = region_mask(x_list_mj,y_list_mj,da_prob,xname='X',yname='Y')

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
        climo_areaperc_file = f'{PREDIR}NMME_MHW_maskEneMean_areaPercEEZ_climo_1991_2020_'
        for nmodel, model in enumerate(model_use_list):
            if model != model_use_list[-1]:
                climo_areaperc_file = climo_areaperc_file+model+'_'
            else:
                climo_areaperc_file = climo_areaperc_file+model+'.nc'

        ds_perc_climo = xr.open_dataset(climo_areaperc_file)
        ds_perc_climo = ds_perc_climo.where(ds_perc_climo!=0)
        # pick the same month in climo file
        climo_month = da_mask_ts.month.data
        da_perc_climo = ds_perc_climo.sel(month=climo_month)
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
        ax2 = fig.add_axes(
            [0,0+GAP*nlead,1,0.5],
            projection=ccrs.PlateCarree(central_longitude=(region_limit[0]+region_limit[1])/2))

        da2 = (da_prob).isel(L=lead)
        if lead == leadtime[0] :
            ax2,im2 = plot_region_map(ax2,da2,clevels=plevel,ytick=False,cmap='inferno')
        else:
            ax2,im2 = plot_region_map(ax2,da2,clevels=plevel,xtick=False,ytick=False,cmap='inferno')
            
        ax2.plot(x_list_mj,y_list_mj,color='lime',
                 linestyle='dashed',markersize=0.0,linewidth=2.1,
                 transform=ccrs.PlateCarree())
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
        ax2 = fig.add_axes(
            [HGAP,0+GAP*nlead,1,0.5],
            projection=ccrs.PlateCarree(central_longitude=(region_limit[0]+region_limit[1])/2))
        da2 = (da_anomoly).isel(L=lead).mean(dim='M').mean(dim='model')
        if lead == leadtime[0] :
            ax2,im2 = plot_region_map(ax2,da2,clevels=mlevel,ytickleft=False,cmap='OrRd')
        else:
            ax2,im2 = plot_region_map(ax2,da2,clevels=mlevel,xtick=False,ytickleft=False,cmap='OrRd')
        ax2.plot(x_list_mj,y_list_mj,color='lime',
                 linestyle='dashed',markersize=0.0,linewidth=2.1,transform=ccrs.PlateCarree())
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
            
    fig.savefig('/home/chsu/mhw_portal/figures/MHW_maps_eez.png', dpi=150, facecolor='w', edgecolor='w',
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
        '/home/chsu/mhw_portal/figures/MHW_area_bar_eez.png',
        dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait',
        transparent=False,
        bbox_inches="tight",
        pad_inches=None
    )
