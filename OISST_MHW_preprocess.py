"""
Preprocessing the OISST data to be dash ready

The 

"""
import os
import warnings
import datetime
import subprocess
from datetime import date
import xesmf as xe
import xarray as xr
import numpy as np
from dask.distributed import Client


### specify date format
dateform = date.today().strftime("%y_%m_%d")   # setup the new output file name (same as date of download)

# get the time of today
today = datetime.datetime.now()
today_year = int(today.year)
today_month = int(today.month)
today_day = int(today.day)

#### setting ####
climo_start_yr = 1991             # determine the climatology/linear trend start year
climo_end_yr = 2020               # determine the climatology/linear trend end year
start_yr = 1991                   # determine the latest yearly output of OISST
# start_yr = 1981                   # determine the latest yearly output of OISST  
latest_yr = today_year            # determine the latest yearly output of OISST  
date = dateform                   # date for the output file name 
mhw_threshold_list = [90,95,99]   # list of percentile to output for mhw threshold
sym_link_latest = True           # whether to symbolic link to the latest netcdf file used in generating the MHW portal data (cron job should be True)


########## Functions ######### 
# Function to calculate the 3 month rolling Quantile
def mj_3mon_quantile(da_data, mhw_threshold=90.):
    
    da_data_quantile = xr.DataArray(coords={'lon':da_data.lon,
                                            'lat':da_data.lat,
                                            'month':np.arange(1,13)},
                                    dims = ['month','lat','lon'])

    for i in range(1,13):
        if i == 1:
            mon_range = [12,1,2]
        elif i == 12 :
            mon_range = [11,12,1]
        else:
            mon_range = [i-1,i,i+1]

        da_data_quantile[i-1,:,:] = (da_data
                                 .where((da_data['time.month'] == mon_range[0])|
                                        (da_data['time.month'] == mon_range[1])|
                                        (da_data['time.month'] == mon_range[2]),drop=True)
                                 .quantile(mhw_threshold*0.01, dim = 'time', skipna = True))

    return da_data_quantile
    
# Function to detrend along time domain
def detrend(da, dim, deg=1):
    # detrend along a single dimension
    coeff = da.polyfit(dim=dim, deg=deg)
    
    return coeff


################# Main program start #############
# start a local cluster
client = Client(n_workers=2,threads_per_worker=100,processes=False)
warnings.simplefilter("ignore")

# Read in daily climatology and daily data
datadir = '/Datasets/noaa.oisst.v2.highres/'
climofile = 'sst.day.mean.ltm.%s-%s.nc'%(climo_start_yr,climo_end_yr)
files = ['sst.day.mean.%i.nc'%year for year in range(start_yr,latest_yr+1)]

# check availability of files (ex: new year data delay)
avai_files = []
for file in files:
    file_path = os.path.join(datadir, file)
    if os.path.exists(file_path):
        avai_files.append(file)
    else:
        print(f"{file} does not exist")

# Read in daily data and resample to monthly
ds_oisst = xr.open_mfdataset([datadir+file for file in avai_files])

data_year = ds_oisst.time.dt.year.data[-1]
data_month = ds_oisst.time.dt.month.data[-1]
data_day = ds_oisst.time.dt.day.data[-1]

da_oisst_mon = ds_oisst.sst.resample(time='MS').mean(dim='time').compute()

# check current data is for current month
if data_year == today_year and data_month == today_month :
    # if it is current month a total 25 days is necessary to generate
    #   a meaningful monthly data
    if today_day < 25 :
        # remove the current month due to insufficiant daily data
        da_oisst_mon = da_oisst_mon.isel(time=slice(None,-1))

# Read in daily climatology and resample to monthly
ds_oisst_climo = xr.open_dataset(datadir+climofile)
ds_oisst_mon_climo = ds_oisst_climo.sst.resample(time='MS').mean(dim='time').compute()

# Read the NMME model output from OpenDAP
#  in order to regrid the OISST to model resolution
model_list = ['GFDL-SPEAR']
forecast_list = ['http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.%s/.FORECAST/.MONTHLY/.sst/dods'%model for model in model_list] 

dict_model_forecast = {}
for nmodel,link in enumerate(forecast_list):
    ds = xr.open_dataset(link,decode_times=False)
    dict_model_forecast[model_list[nmodel]] = ds.rename({'X':'lon','Y':'lat'})

# Regrid the OISST data based on NMME model
print('--- creating regridder ---')
regridder = xe.Regridder(da_oisst_mon, 
                         dict_model_forecast[model_list[0]],
                         'bilinear',
                         periodic=True,
                         unmapped_to_nan=True)          

# 2D interpolation
da_oisst_mon = regridder(da_oisst_mon)
ds_oisst_mon_climo = regridder(ds_oisst_mon_climo)

# # output intermediate data
# ds = xr.Dataset()
# ds['sst'] = da_oisst_mon
# ds.to_netcdf('/scratch/chsu/MHW_dash/oisst.mon.mean.nc')

# ds = xr.Dataset()
# ds['sst'] = ds_oisst_mon_climo
# ds.to_netcdf('/scratch/chsu/MHW_dash/oisst.mon.ltm.nc')

# calulate monthly anomaly of OISST
da_oisst_mon_anom = da_oisst_mon.groupby('time.month')-ds_oisst_mon_climo.groupby('time.month').mean()

# detrending the anomaly based on the linear trend during the climatology period 
da_p = detrend(da_oisst_mon_anom.where((da_oisst_mon_anom['time.year']>=climo_start_yr)&
                                       (da_oisst_mon_anom['time.year']<=climo_end_yr),
                                       drop=True),
               'time',
               deg=1)
da_oisst_mon_anom_detrend = da_oisst_mon_anom-xr.polyval(da_oisst_mon_anom['time'], da_p.polyfit_coefficients)

# output anomaly
out_dir = '/Datasets.private/marinehw/'

# command_line = "rm %soisst.mon.anom.nc"%(out_dir)
# print(command_line)
# p = subprocess.call(command_line,
#                     shell=True,
#                     executable="/usr/bin/bash")
# command_line = "rm %soisst.mon.detrend.anom.nc"%(out_dir)
# print(command_line)
# p = subprocess.call(command_line,
#                     shell=True,
#                     executable="/usr/bin/bash")

da_oisst_mon_anom = da_oisst_mon_anom.rename('sst_anom')
try:
    da_oisst_mon_anom.to_netcdf(f'{out_dir}oisst.mon.anom_{date}.nc')
except PermissionError:
    p = subprocess.call(f'rm -f {out_dir}oisst.mon.anom_{date}.nc',
                    shell=True,
                    executable="/usr/bin/bash")
    da_oisst_mon_anom.to_netcdf(f'{out_dir}oisst.mon.anom_{date}.nc')

da_oisst_mon_anom_detrend = da_oisst_mon_anom_detrend.rename('sst_anom')
try:
    da_oisst_mon_anom_detrend.to_netcdf(f'{out_dir}oisst.mon.detrend.anom_{date}.nc')
except PermissionError:
    p = subprocess.call(f'rm -f {out_dir}oisst.mon.detrend.anom_{date}.nc',
                    shell=True,
                    executable="/usr/bin/bash")
    da_oisst_mon_anom_detrend.to_netcdf(f'{out_dir}oisst.mon.detrend.anom_{date}.nc')

if sym_link_latest:
    # create latest file
    command_line = f"ln -sf {out_dir}oisst.mon.anom_{date}.nc {out_dir}oisst.mon.anom_latest.nc"
    print(command_line)
    p = subprocess.call(command_line,
                        shell=True,
                        executable="/usr/bin/bash")

    command_line = f"ln -sf {out_dir}oisst.mon.detrend.anom_{date}.nc {out_dir}oisst.mon.detrend.anom_latest.nc"
    print(command_line)
    p = subprocess.call(command_line,
                        shell=True,
                        executable="/usr/bin/bash")

# da_oisst_mon_anom.to_netcdf(out_dir+'oisst.mon.anom_latest.nc')
# da_oisst_mon_anom_detrend.to_netcdf(out_dir+'oisst.mon.detrend.anom_latest.nc')

for threshold in mhw_threshold_list :
    # calculate quantile for detrend and with trend 
    da_oisst_mon_quantile = mj_3mon_quantile(da_oisst_mon_anom, mhw_threshold=threshold)
    da_oisst_mon_detrend_quantile = mj_3mon_quantile(da_oisst_mon_anom_detrend, mhw_threshold=threshold)

    # # using monthly quantile to identify MHW
    # da_oisst_mon_mhw = da_oisst_mon_anom.where(da_oisst_mon_anom.groupby('time.month')>da_oisst_mon_quantile)
    # da_oisst_mon_mhw_detrend = da_oisst_mon_anom_detrend.where(da_oisst_mon_anom_detrend.groupby('time.month')>da_oisst_mon_detrend_quantile)

    # Xarray naming the dataarray for output
    # da_oisst_mon_mhw = da_oisst_mon_mhw.rename('mhw')
    da_oisst_mon_quantile = da_oisst_mon_quantile.rename('sst_anom')
    # da_oisst_mon_mhw_detrend = da_oisst_mon_mhw_detrend.rename('mhw')
    da_oisst_mon_detrend_quantile = da_oisst_mon_detrend_quantile.rename('sst_anom')

    # command_line = "rm %soisst.mon.quantile%i.nc"%(out_dir,threshold)
    # print(command_line)
    # p = subprocess.call(command_line,
    #                     shell=True,
    #                     executable="/usr/bin/bash")
    # command_line = "rm %soisst.mon.detrend.quantile%i.nc"%(out_dir,threshold)
    # print(command_line)
    # p = subprocess.call(command_line,
    #                     shell=True,
    #                     executable="/usr/bin/bash")
    
    # output to netcdf file
    # da_oisst_mon_mhw.to_netcdf(out_dir+'oisst.mon.mhw.nc')
    try:
        da_oisst_mon_quantile.to_netcdf(f'{out_dir}oisst.mon.quantile{threshold}_{date}.nc')
    except PermissionError:
        p = subprocess.call(f'rm -f {out_dir}oisst.mon.quantile{threshold}_{date}.nc',
                        shell=True,
                        executable="/usr/bin/bash")
        da_oisst_mon_quantile.to_netcdf(f'{out_dir}oisst.mon.quantile{threshold}_{date}.nc')
    # da_oisst_mon_mhw_detrend.to_netcdf(out_dir+'oisst.mon.detrend.mhw.nc')
    try:
        da_oisst_mon_detrend_quantile.to_netcdf(f'{out_dir}oisst.mon.detrend.quantile{threshold}_{date}.nc')
    except PermissionError:
        p = subprocess.call(f'rm -f {out_dir}oisst.mon.detrend.quantile{threshold}_{date}.nc',
                        shell=True,
                        executable="/usr/bin/bash")
        da_oisst_mon_detrend_quantile.to_netcdf(f'{out_dir}oisst.mon.detrend.quantile{threshold}_{date}.nc')

    if sym_link_latest:
        # create latest file
        command_line = f"ln -sf {out_dir}oisst.mon.quantile{threshold}_{date}.nc {out_dir}oisst.mon.quantile{threshold}_latest.nc"
        print(command_line)
        p = subprocess.call(command_line,
                            shell=True,
                            executable="/usr/bin/bash")
        command_line = f"ln -sf {out_dir}oisst.mon.detrend.quantile{threshold}_{date}.nc {out_dir}oisst.mon.detrend.quantile{threshold}_latest.nc"
        print(command_line)
        p = subprocess.call(command_line,
                            shell=True,
                            executable="/usr/bin/bash")
        
        # da_oisst_mon_quantile.to_netcdf(out_dir+f'oisst.mon.quantile{threshold}_latest.nc')
        # da_oisst_mon_detrend_quantile.to_netcdf(out_dir+f'oisst.mon.detrend.quantile{threshold}_latest.nc')
