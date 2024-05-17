# %%
"""
# NMME Marine Heat Wave
 
The script calculate the marine heat wave event based on 
Mike Jacox et al., [2022]

Using NMME model hindcast and forecast obtained from 
http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
with variable SST

The steps are described below.
------------------------------
1. calculate monthly ensemble mean climatology for each model initialization month and lead time
2. calculate the anomaly for each model by subtracting the climatology
3. calculate the threshold based on the anomaly (fixed at 90 quantile for now)
4. calculate the events that are higher than the threshold

"""


# %%
# start a local cluster
import sys
import warnings
from typing import Tuple
import xarray as xr
from dask.distributed import Client
from nmme_climo import read_nmme
from mhw_nmme_threshold import nmme_3mon_quantile
from nmme_download import iri_nmme_models

warnings.simplefilter("ignore")

def detrend(
        da: xr.DataArray,
        dim: str,
        deg: int=1
) -> Tuple[xr.DataArray,xr.DataArray] :
    """detrending along `dim` with set `deg` for polynomial fit

    Parameters
    ----------
    da : xr.DataArray
        The dataarray/variable one want to do linear detrend along `dim`
    dim : str
        dimension name where the linear detrending is performed
    deg : int, optional
        The linear detrend is tested. Setting to 2 will perform 2nd order
        polynomial fitting and remove the linear and quadrature term,
        by default 1

    Returns
    -------
    Tuple[xr.DataArray,xr.DataArray]
        Output include two xr.DataArray object. First one is 
        the detrended `da` and second one is the polynomial
        coeff. saved for detrending future dataset.
    """
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg, skipna=True).compute()
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    da_detrend = (da - fit).compute()
    return da_detrend, p

def cal_anom(data_path,model,syear,fyear,remove_trend=False,climo_file=None,polyfit_file=None):
    """
    calculate the anomaly of the data
    """

    ds_model = read_nmme(
        forecast_files = data_path,
        model = model
    )

    da_model = ds_model['sst']
    da_model = da_model.where((da_model['S.year']>=syear)&
                              (da_model['S.year']<=fyear),drop=True)

    print('read climatology')
    da_ensmean_climo = xr.open_dataset(climo_file,chunks={'S':1,'L':1})['sst']

    print('calculating anomaly')
    da_anomaly = da_model.groupby('S.month') - da_ensmean_climo

    if not remove_trend :
        return da_anomaly.compute()
    else:
        print('calculate trend and detrended')
        da_anomaly_detrend, da_p = detrend(da_anomaly, 'S', deg=1)
        da_p.to_netcdf(polyfit_file)

        return da_anomaly_detrend

def cal_mhw(da_anomaly,mhw_thres,output_threshold_path,output_mhw_path):
    """
    calculate the related marine heatwave based on the 
    set threshold in argument `mhw_thres`
    """
    ds_mhw_threshold = xr.Dataset()
    ds_mhw = xr.Dataset()

    print(f'calculating threshold {mhw_thres}')
    da_threshold = nmme_3mon_quantile(da_anomaly, mhw_threshold=mhw_thres)
    ds_mhw_threshold[f'threshold_{mhw_thres}'] = da_threshold

    print(f'calculating MHW {mhw_thres}')
    da_mhw = da_anomaly.where(
        da_anomaly.groupby('S.month')>=ds_mhw_threshold[f'threshold_{mhw_thres}']
    )
    ds_mhw[f'mhw_{mhw_thres}'] = da_mhw

    print('file output')
    ds_mhw_threshold.to_netcdf(output_threshold_path)
    ds_mhw.to_netcdf(output_mhw_path)


# %%
if __name__ == "__main__" :

    client = Client(processes=False)
    print(client)
    print(client.cluster.dashboard_link)

    BASEDIR='/Datasets.private/marinehw/nmme_sst_raw/'
    PROCDIR='/Datasets.private/marinehw/NMME_preprocess/'
    if len(sys.argv) < 2:
        print("Usage: python nmme_mhw_detrend.py <model name>")

    dict_model = iri_nmme_models()
    avai_model_list = list(dict_model.keys())

    # read user input
    model_list = []
    for modelname in sys.argv[1:]:
        if modelname in avai_model_list:
            model_list.append(str(modelname))
            print(f'{modelname} exist, add to process list')
        else:
            print(f'{modelname} does not exist')
            print(f'Make sure pick model from {avai_model_list}')

        # construct model list
        forecast_files = f'{BASEDIR}{modelname}_forecast_??_??_??_??????.nc'
        climo_file0 = f'{PROCDIR}{modelname}_climo.nc'
        output_file0 = f'{PROCDIR}{modelname}_polyfit_p.nc'
        output_file1 = f'{PROCDIR}{modelname}_threshold.nc'
        output_file2 = f'{PROCDIR}{modelname}_mhw.nc'

        # consistent threshold period
        START_YEAR = 1991
        END_YEAR = 2020

        # three fixed threshold output
        mhw_threshold = [90]

        print('-------------')
        print(modelname)
        print('-------------')

        da_anom = cal_anom(
            forecast_files,
            modelname,
            START_YEAR,
            END_YEAR,
            remove_trend=True,
            climo_file=climo_file0,
            polyfit_file=output_file0
        )

        for m in mhw_threshold:
            output1 = output_file1[:-3]+f'{m}.nc'
            output2 = output_file2[:-3]+f'{m}.nc'
            cal_mhw(
                da_anom,
                m,
                output1,
                output2
            )
            print(output1)
            print(output2)
