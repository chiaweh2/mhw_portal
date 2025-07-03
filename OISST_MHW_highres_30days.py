"""
Calculate the past 30 days mean MHW

Daily update is available if the oisst data is updated.
"""
import warnings
# import argparse
import datetime
# import subprocess
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from dask.distributed import Client


#################### Functions ##################

def detrend(da_var, dim, deg=1):
    """ 
    Detrend the xarray.DataArray along a single dimension 
    with different degree of least square polynomial fitting.

    Parameters
    ----------
    da_var : xarray.DataArray 
        The data array one want to calculate the linear trend.
	dim : string  
        Dimension name in the DataArray one wish to calculate the 
        linear trend.
    deg : integer  
        The degree of least square fitting one wish to perform. 
        Default is set to degree 1.

    Returns
    -------
    p_coeff : xarray.DataArray
        The least square fitting result in the form of coefficients
        in the polynomial.

    Raises
    ------

    """
    p_coeff = da_var.polyfit(dim=dim, deg=deg)

    return p_coeff


def rolling_window_days(today,data_syear):
    """ 
    Finding the date range for 
    - past 30 days
    - previous 90 days windows in all previous years

    Parameters
    ----------
    today : datetime.datetime
        The datetime object used to determine the 
        - past 30 days range
        - previous 90 days windows in all previous years

    data_syear : integer
        The start year of the data one wants to use

    Returns
    -------
    previous_90days : list
        A list of all previous 90 day window date
    past_30days : list
        A list of all date in the past 30 days

    Raises
    ------

    """
    end_date = today+datetime.timedelta(days=30)
    end_date_year = end_date.year
    end_date_month = end_date.month
    end_date_day = end_date.day

    if end_date_month == 2 and end_date_day == 29:
        end_date = end_date-datetime.timedelta(days=1)
        end_date_year = end_date.year
        end_date_month = end_date.month
        end_date_day = end_date.day

    dyear = (end_date_year-data_syear)+1

    previous_90days = []
    for dy in range(dyear-1,0,-1):
        enddate = datetime.datetime(end_date_year-dy, end_date_month, end_date_day)
        previous_90days.append(
            xr.cftime_range(end=enddate, periods=90, freq='D',calendar='standard')
        )

    past_30days = xr.cftime_range(end=today, periods=30, freq='D',calendar='standard')

    return previous_90days, past_30days

def three_30days_mean(da_var,previous_90days):
    """ 
    calculate the 3 consecutive 30 days mean of a DataArray 
    based on the provided list of 90 days window

    Parameters
    ----------
    da_var : Xarray.DataArray
        The DataArray with time dimension which need the 
        30 days mean in the 90 day windows.

    previous_90days : list of list of 90 element datetime/cftime object 
        A list of 90 day windows. Each 90 day window is a list of 90
        datetime/cftime object.

    Returns
    -------
    da_var_30daymean : Xarray.DataArray
        The DataArray with time dimension of 30 day means that includes 
        in the 90 day windows.

    Raises
    ------

    """
    templist = []
    for date_range in previous_90days:
        templist.append(da_var.sel(time=date_range[0:30],method='nearest').mean(dim='time'))
        templist.append(da_var.sel(time=date_range[30:60],method='nearest').mean(dim='time'))
        templist.append(da_var.sel(time=date_range[60:90],method='nearest').mean(dim='time'))
    da_var_30daymean = xr.concat(templist,dim='time')

    return da_var_30daymean

def orthographic_us(da_var,text="With trend"):
    """
    Plotting function for mhw near US at orthographic projection
    """
    level = np.arange(0,6,1)

    fig = plt.figure(2,figsize=(15,10))
    ax2 = fig.add_axes(
        [0,0,1,0.5],
        projection=ccrs.Orthographic(central_longitude=250, central_latitude=30.0)
    )

    im = da_var.plot.pcolormesh(
        x='lon',
        y='lat',
        ax=ax2,
        levels=level,
        extend='both',
        cmap='OrRd',
        transform=ccrs.PlateCarree(central_longitude=0.)
    )

    cb=im.colorbar
    cb.remove()

    cbaxes=fig.add_axes([0.46,0-0.02,0.35,0.01])
    cbar=fig.colorbar(im,cax=cbaxes,orientation='horizontal')
    cbar.set_ticks(level)
    cbar.set_ticklabels([f"{int(n)}" for n in level])
    cbar.ax.tick_params(labelsize=12,rotation=0)
    cbar.set_label(label='Past 30 Days Marine Heatwave Magnitude ($^o$C)',size=12, labelpad=15)

    ax2.set_global()
    # global_extent = ax2.get_extent(crs=ccrs.PlateCarree())
    # ax2.set_extent((-160, -40,20, 80), crs=ccrs.PlateCarree())

    ax2.coastlines(resolution='110m',linewidths=1)

    land = cfeature.NaturalEarthFeature('physical', 'land', '110m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'])

    # states_provinces = cfeature.NaturalEarthFeature(
    #         category='cultural',
    #         name='admin_1_states_provinces_lines',
    #         scale='110m',
    #         facecolor='none')

    ax2.add_feature(land,color='lightgrey',linewidths=1)
    # ax2.add_feature(states_provinces,edgecolor='grey',linewidths=1)
    ax2.add_feature(cfeature.BORDERS,linewidths=0.1)
    ax2.text(0.73, 0.01, text, fontsize=15, transform=ax2.transAxes)
    ax2.set_title("")

    plt.close(fig)

    return fig

def robinson_global(da_var,dtime,text='with trend'):
    """
    Plotting function for mhw over the glpobe at robinson projection
    """
    level = np.arange(0,6,1)

    fig = plt.figure(2,figsize=(15,10))
    ax2 = fig.add_axes([0,0,1,0.5],projection=ccrs.Robinson(central_longitude=210))

    im = da_var.plot.pcolormesh(
        x='lon',
        y='lat',
        ax=ax2,
        levels=level,
        extend='both',
        cmap='OrRd',
        transform=ccrs.PlateCarree(central_longitude=0.)
    )
    cb=im.colorbar
    cb.remove()

    cbaxes=fig.add_axes([0.3,0-0.03,0.35,0.01])
    cbar=fig.colorbar(im,cax=cbaxes,orientation='horizontal')
    cbar.set_ticks(level)
    cbar.set_ticklabels([f"{int(n)}" for n in level])
    cbar.ax.tick_params(labelsize=16,rotation=0)
    cbar.set_label(label='Past 30 Days Marine Heatwave Magnitude ($^o$C)',size=16, labelpad=15)

    ax2.set_global()
    ax2.coastlines(resolution='110m',linewidths=1)

    land = cfeature.NaturalEarthFeature('physical', 'land', '110m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['land'])
    ax2.add_feature(land,color='lightgrey')
    ax2.set_title(f"OISSTv2 highres {text}",fontsize=20,pad=10)

    ax2.text(0.8,
            -0.06,
            f"[{dtime[0]:%B-%d-%Y} to {dtime[-1]:%B-%d-%Y}]",
            fontsize=12,
            transform=ax2.transAxes)

    plt.close(fig)

    return fig


if __name__ == '__main__':

    # # Create the argument parser
    # parser = argparse.ArgumentParser()

    # # Add the kwarg argument
    # parser.add_argument('--webinstall')

    # # Parse the command-line arguments
    # args = parser.parse_args()

    # # Access the kwarg value
    # WINSTALL = args.webinstall

    # if WINSTALL.lower() == 'true':
    #     WINSTALL = True
    # elif WINSTALL.lower() == 'false':
    #     WINSTALL = False
    # else:
    #     raise ValueError("Invalid boolean string (please enter 'true' or 'false')")
    # # WINSTALL = False

    # start a local cluster
    client = Client(processes=False)
    warnings.simplefilter("ignore")

    # marine heatwaves percentile threshold
    MHW_THRESHOLD = 90

    # use the oisst dataset start from set year
    START_YEAR = 1991

    # climatology and linear trend period
    CLIMO_START_YR = 1991             # determine the climatology/linear trend start year
    CLIMO_END_YR = 2020               # determine the climatology/linear trend end year

    # determine the date range and dates
    current = datetime.datetime.now()
    date_ranges, date_past30 = rolling_window_days(current,START_YEAR)

    # Read in daily climatology and daily data
    DATADIR = '/Datasets/noaa.oisst.v2.highres/'
    CLIMOFILE = f'sst.day.mean.ltm.{CLIMO_START_YR}-{CLIMO_END_YR}.nc'
    files = [f'sst.day.mean.{year}.nc' for year in range(START_YEAR,current.year+1)]

    ds_oisst = xr.open_mfdataset([DATADIR+file for file in files])
    ds_oisst_climo = xr.open_dataset(DATADIR+CLIMOFILE)

    # calulate sst daily anomaly
    print('calculate daily anomaly (.compute())')
    da_oisst_anom = (
        ds_oisst.sst.groupby('time.dayofyear')-
        ds_oisst_climo.sst.groupby('time.dayofyear').mean()
    ).compute()

    # detrending the anomaly based on the linear trend during the climatology period
    print('calculate detrend coeff.')
    da_p = detrend(
        da_oisst_anom
            .where((da_oisst_anom['time.year']>=CLIMO_START_YR)&
                   (da_oisst_anom['time.year']<=CLIMO_END_YR),drop=True),
        'time',
        deg=1
    )
    print('detrend')
    da_oisst_anom_detrend = (
        da_oisst_anom-xr.polyval(da_oisst_anom['time'], da_p.polyfit_coefficients)
    )

    # calculate the 3 consecutive 30-day mean in each 90-day window
    print('calculate 3 consecutive 30-day mean')
    da_oisst_anom_3mon = three_30days_mean(da_oisst_anom,date_ranges)
    da_oisst_anom_detrend_3mon = three_30days_mean(da_oisst_anom_detrend,date_ranges)

    # calculate the MHW threshold based on percentile
    print('calculate quantile')
    da_data_quantile = da_oisst_anom_3mon.quantile(MHW_THRESHOLD*0.01, dim='time', skipna=True)
    da_data_detrend_quantile = (
        da_oisst_anom_detrend_3mon
        .quantile(MHW_THRESHOLD*0.01, dim='time', skipna=True)
    )

    # calculate the past 30 day mean
    print('calculate past 30-day mean')
    da_oisst_anom_past30 = da_oisst_anom.sel(time=date_past30,method='nearest').mean(dim='time')
    da_oisst_anom_detrend_past30 = (
        da_oisst_anom_detrend.sel(time=date_past30,method='nearest').mean(dim='time')
    )

    # calculate marine heatwaves magnitude
    print('calculate MHW')
    da_mhw = da_oisst_anom_past30.where(da_oisst_anom_past30>da_data_quantile)
    da_mhw_detrend = (
        da_oisst_anom_detrend_past30
        .where(da_oisst_anom_detrend_past30>da_data_detrend_quantile)
    )

    ######################## plotting US ###########################
    print('plotting us with trend')
    fig1 = orthographic_us(da_mhw,text="")
    fig1.savefig('/httpd-test/psd/marine-heatwaves/img/MHW_maps_demo_trend_us.png',
                dpi=300,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                format=None,
                transparent=False,
                bbox_inches="tight",
                pad_inches=None
    )
    print('plotting us without trend')
    fig2 = orthographic_us(da_mhw_detrend,text="")
    fig2.savefig('/httpd-test/psd/marine-heatwaves/img/MHW_maps_demo_detrend_us.png',
                dpi=300,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                format=None,
                transparent=False,
                bbox_inches="tight",
                pad_inches=None
    )

    ###################### plotting global ######################
    print('plotting global with trend')
    fig3 = robinson_global(da_mhw,date_past30,text='')
    fig3.savefig('/httpd-test/psd/marine-heatwaves/img/MHW_maps_demo_trend_global.png',
                dpi=300,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                format=None,
                transparent=False,
                bbox_inches="tight",
                pad_inches=None
    )
    print('plotting global without trend')
    fig4 = robinson_global(da_mhw_detrend,date_past30,text='')
    fig4.savefig('/httpd-test/psd/marine-heatwaves/img/MHW_maps_demo_detrend_global.png',
                dpi=300,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                format=None,
                transparent=False,
                bbox_inches="tight",
                pad_inches=None
    )

    # outpu netcdf file
    ds_mhw_30days = xr.Dataset()
    da_mhw.attrs['units'] = 'degree Celcius'
    da_mhw.attrs['long_name'] = 'marine heatwave magnitude in the past 30 days with long-term trend included'
    ds_mhw_30days['mhw'] = da_mhw
    da_mhw_detrend.attrs['units'] = 'degree Celcius'
    da_mhw_detrend.attrs['long_name'] = 'marine heatwave magnitude in the past 30 days with long-term trend removed'
    ds_mhw_30days['mhw_detrend'] = da_mhw_detrend
    ds_mhw_30days.attrs['title'] = 'Past 30 Days Marine Heatwave Magnitude'
    ds_mhw_30days.attrs['reference'] = 'Derived at NOAA Physical Science Laboratory'
    ds_mhw_30days.attrs['description'] = (
    'The magnitude of marine heatwaves (MHW) in the past 30 days is '+
    'calculated using the daily OISSTv2 high-resolution sea surface temperature (SST) '+
    'dataset hosted at PSL. You can access the dataset at PSL data server. '+
    'The magnitude represents the 30-day mean of the daily SST anomaly. '+
    'The daily climatology and linear trend is based on the period of 1991-2020. '+
    'To establish the threshold for identifying MHW events, we adopt a method inspired '+
    'by the 3-month window approach described in the study by Jacox et al.. '+
    'As the "past 30 days" might not fall within the same month, we employ a '+
    'compromise by using a 90-day window instead. For each year going back to 1991, '+
    'we calculate three consecutive 30-day means within this 90-day window. '+
    'The resulting 30-day means are then utilized to determine the MHW threshold, '+
    'which corresponds to the 90th percentile. For example, on June 30, 2023, '+
    'the past 30-day mean will encompass the period from June 1, 2023, to June 30, 2023. '+
    'For each previous year, the 90-day window will consist of three consecutive 30-day '+
    'means spanning from May 2 to May 31, June 1 to June 30, and July 1 to July 30. '+
    'Based on the threshold determined using the 90-day window approach, all 30-day '+
    'mean SST anomalies that are higher than the threshold are shown on the map, '+
    'indicating the areas experiencing marine heatwaves and the corresponding magnitudes.'
    )
    ds_mhw_30days.attrs['contact'] = 'psl.marineheatwaves@noaa.gov'
    ds_mhw_30days.attrs['dataset'] = 'OISSTv2 https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html'
    dtime = date_past30
    ds_mhw_30days.attrs['date_range'] = f"{dtime[0]:%B-%d-%Y} to {dtime[-1]:%B-%d-%Y}"

    ds_mhw_30days.to_netcdf('/Public/chsu/share_mhw/mhw_30days.nc')


    # # webinstall all new images
    # print('webinstall or not')
    # if WINSTALL :
    #     p = subprocess.call("webinstall -m dailyOISSTmhw /httpd-test/psd/marine-heatwaves/img/",
    #                         shell=True,
    #                         executable="/usr/bin/bash")
