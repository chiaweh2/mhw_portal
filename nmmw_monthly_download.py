"""
This script is to perform monthly downloaded NMME model 
from IRI to the new unified folder. 
/Datasets.private/marinehw/nmme_sst_raw

Only the newest forecast will be download!!!
"""

import datetime
import glob
from datetime import date
import cftime
import numpy as np
import xarray as xr
from dask.distributed import Client
from nmme_download import iri_nmme_models


if __name__=="__main__":
    client = Client(processes=False)

    # get the time of today
    today = datetime.datetime.now()
    today_year = int(today.year)
    today_month = int(today.month)
    today_day = int(today.day)

    today = date.today()
    dateform = today.strftime("%y_%m_%d")

    # get all hindcast and forecast IRI OPeNDAP URL
    NMME_IRI_LOC = iri_nmme_models()
    OUTPUTDIR = '/Datasets.private/marinehw/nmme_sst_raw/'

    dict_model_forecast = {}
    for name,links in NMME_IRI_LOC.items():
        print('=============')
        print(f'downloading {name}')
        link = links[-1]  # using the last link that is forecast

        # forecast opendap lazy loaded
        dict_model_forecast[name] = xr.open_dataset(
            link,
            chunks={'M':1,'L':1,'S':1},
            decode_times=False
        )

        # find on prem storage availability
        on_disc_list = glob.glob(
            f'{OUTPUTDIR}'+
            f'{name}_forecast_??_??_??_??????.nc'
        )
        ds_forecast_old = xr.open_mfdataset(on_disc_list,decode_times=False)
        ds_forecast_old['S'] = cftime.num2date(
            ds_forecast_old.S.values,
            ds_forecast_old.S.units,
            calendar='360_day'
        )
        ds_forecast_old = ds_forecast_old.isel(S=slice(-1,None))

        # find IRI availability
        ds_forecast = dict_model_forecast[name].copy()
        ds_forecast['S'] = cftime.num2date(
            dict_model_forecast[name].S.values,
            dict_model_forecast[name].S.units,
            calendar='360_day'
        )

        # find last day on prem
        last_date = ds_forecast_old.S.isel(S=-1).data
        # find last day on prem in the IRI storage index
        xind, = np.where(ds_forecast.S.data==last_date)

        # test data availability
        try:
            ds_new = ds_forecast.isel(S=xind[0]+1)
        except IndexError:
            print('new data not available')
            continue

        # getting only the forecast that does not exist on prem
        # - for date
        ds_forecast = ds_forecast.isel(S=slice(xind[0]+1,None))
        # - for origianl data format
        dict_model_forecast[name] = (
            dict_model_forecast[name]
            .isel(S=slice(xind[0]+1,None))
            .load()
        )

        for s,S in enumerate(ds_forecast.S.data):
            print(f'downloading {S}')
            dyear = ds_forecast.isel(S=s).S.dt.year.data
            dmonth = ds_forecast.isel(S=s).S.dt.month.data
            dict_model_forecast[name].isel(S=[s]).to_netcdf(
                f'{OUTPUTDIR}'+
                f'{name}_forecast_{dateform}_{dyear:04d}{dmonth:02d}.nc'
            )
