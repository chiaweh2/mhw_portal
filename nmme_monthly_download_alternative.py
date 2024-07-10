"""
This script is to perform alternative monthly downloaded NMME model 
from IRI to the new unified folder. when the gereral way is not working
/Datasets.private/marinehw/nmme_sst_raw

Only the newest forecast will be download!!!
"""

import datetime
import glob
import cftime
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

    # date form (download stamp)
    today = datetime.date.today()
    dateform = today.strftime("%y_%m_%d")

    # get all hindcast and forecast IRI OPeNDAP URL
    NMME_IRI_LOC = iri_nmme_models()
    OUTPUTDIR = '/Datasets.private/marinehw/nmme_sst_raw/'

    dict_model_forecast = {}
    for name,links in NMME_IRI_LOC.items():
        print('=============')
        print(f'checking {name}')
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
        ds_forecast_old = xr.open_mfdataset(
            on_disc_list,
            decode_times=False,
            concat_dim='S',
            combine='nested'
        )
        ds_forecast_old = ds_forecast_old.sortby('S')
        s = ds_forecast_old.S.data[-1]+1

        ds_forecast_old['S'] = cftime.num2date(
            ds_forecast_old.S.values,
            ds_forecast_old.S.units,
            calendar='360_day'
        )
        
        Mdim = ds_forecast_old.M.data
        Ldim = ds_forecast_old.L.data
        ds_list_m = []
        for m in Mdim:
            ds_list_l = []
            for l in Ldim:
                sublink = f'{link[:-4]}M/{m:0.1f}/VALUE/L/{l:0.1f}/VALUE/S/{s:0.1f}/VALUE/dods'
                ds_temp = xr.open_dataset(
                    sublink,
                    chunks={'M':1,'L':1,'S':1},
                    decode_times=False
                )
                if s == ds_temp.S.data:
                    ds_list_l.append(ds_temp)
                else:
                    pass
            try:
                ds_list_m.append(xr.concat(ds_list_l,dim='L'))
            except ValueError:
                pass
        try:
            ds_forecast = xr.concat(ds_list_m,dim='M')
        except ValueError:
            print('new data not available')
            continue

        forecast_cftime = cftime.num2date(
            ds_forecast.S.values,
            ds_forecast.S.units,
            calendar='360_day'
        )

        for s,S in enumerate(forecast_cftime):
            print(f'downloading {S}')
            dyear = S.year
            dmonth = S.month
            ds_forecast.to_netcdf(
                f'{OUTPUTDIR}'+
                f'{name}_forecast_{dateform}_{dyear:04d}{dmonth:02d}.nc'
            )
