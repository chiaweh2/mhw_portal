import numpy as np
import xarray as xr
from numba import jit


@jit(nopython=True)
def git_total_quantile(
        data: np.array,
        data_months: np.array,
        total_threshold: float = 75.
    ) ->  np.array :
    """
    This function is designed for calculating the nmme 
    TOTAL threshold (time series - area weighted time series). 
    
    The threshold is calculated using a monthly value to 
    determine the X quantile value for each month.

    !!! jitting it !!! (making sure the numba support from Numba)

    Parameters
    ----------
    data : np.array
        numpy array that its spread at each grid point 
        is used to determine the marine heatwave threshold.
        dimension should be [S,L]

    data_months : np.array
        numpy array that has the month information of the 
        data array above (same length as 'S' and 1 dimension).

    total_threshold : float
        the percentile number used to define the TOTAL index 
        threshold

    Returns
    -------
    quantile : np.array
        A numpy array that contains the marine heatwave
        threshold at each grid point for each month

    Raises
    ------
    """
    quantile_size = (int(12),int(data.shape[1])) # need to be tuple
    quantile = np.zeros(quantile_size)+np.nan

    for i in range(1,13):
        month = i
        ind, = np.where(data_months == month)

        # only support first two arguments in np.quantile for numba
        #  https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#calculation
        #  looping two dimension
        for j in range(data.shape[1]):
            quantile[i-1,j] = np.quantile(
                data[ind,j],
                total_threshold*0.01
            )

    return quantile

def nmme_total_quantile(da_data, total_threshold=75):
    """
    This function is designed for calculating the nmme 
    TOTAL threshold (time series - area weighted time series). 
    
    The threshold is calculated using a monthly value to 
    determine the X quantile value for each month.

    Parameters
    ----------
    da_data : xr.DataArray 
        For TOTAL calculation, it should be the area weighted sst anomaly
        of a previous 6 months rolling mean DataArray. 
        Since this function is designed for NMME model. The DataArray need 
        to be in the format of NMME model output.
    total_threshold : float
        the X quantile one wants to calculate. 

    Returns
    -------
    da_data_quantile : xr.DataArray
        The xr.DataArray for the quantile 

    Raises
    ------

    """

    da_data_stacked = da_data.stack(allens=('S','M')).compute()
    data = da_data_stacked.data
    months = da_data_stacked.month.data   # same length as S
    quantile = git_total_quantile(
        data=data,
        data_months=months,
        total_threshold=total_threshold
    )

    da_data_quantile = xr.DataArray(
        data=quantile,
        coords={
            'month':np.arange(1,13),
            'L':da_data.L
        },
        dims = ['month','L']
    )

    return da_data_quantile


def nmme_1mon_ts_quantile(da_data, total_threshold=75):
    """
    This function is designed for calculating the nmme 
    TOTAL threshold (time series - area weighted time series). 
    
    The threshold is calculated using a monthly value to 
    determine the X quantile value for each month.

    Parameters
    ----------
    da_data : xr.DataArray 
        For TOTAL calculation, it should be the area weighted sst anomaly
        of a previous 6 months rolling mean DataArray. 
        Since this function is designed for NMME model. The DataArray need 
        to be in the format of NMME model output.
    total_threshold : float
        the X quantile one wants to calculate. 

    Returns
    -------
    da_data_quantile : xr.DataArray
        The xr.DataArray for the quantile 

    Raises
    ------

    """
    
    da_data_quantile = xr.DataArray(
        coords={
            'month':np.arange(1,13),
            'L':da_data.L
        },
        dims = ['month','L']
    )

    for i in range(1,13):
        if i == 1:
            mon_range = [1]
        elif i == 12 :
            mon_range = [12]
        else:
            mon_range = [i]

        da_data_quantile[i-1,:] = (
            da_data
            .where(da_data['S.month'] == mon_range[0],drop=True)
            .stack(allens=('S','M'))
            .compute()
            .quantile(total_threshold*0.01, dim = 'allens', method='linear',skipna = True)
        )

    return da_data_quantile


def nmme_1mon_quantile(da_data, mhw_threshold=90.):
    """
    This function is designed for calculating the nmme 
    marine heat wave threshold.
    
    The threshold is calculated using a 3 month window
    to identified the X quantile value for each month.

    Parameters
    ----------
    da_data : xr.DataArray 
        For marine heat wave calculation, it should be the sst DataArray. 
        Since this function is designed for NMME model. The DataArray need 
        to be in the format of NMME model output.
    mhw_threshold : float
        the X quantile one wants to calculate. 

    Returns
    -------
    da_data_quantile : xr.DataArray
        The xr.DataArray for the quantile 

    Raises
    ------

    """
    
    
    da_data_quantile = xr.DataArray(coords={'X':da_data.X,
                                            'Y':da_data.Y,
                                            'month':np.arange(1,13),
                                            'L':da_data.L},
                                    dims = ['month','L','Y','X'])

    for i in range(1,13):
        if i == 1:
            mon_range = [1]
        elif i == 12 :
            mon_range = [12]
        else:
            mon_range = [i]

        da_data_quantile[i-1,:,:,:] = (da_data
                                 .where(da_data['S.month'] == mon_range[0],drop=True)
                                 .stack(allens=('S','M'))
                                 .quantile(mhw_threshold*0.01, dim = 'allens', method='linear',skipna = True))

    return da_data_quantile


def nmme_3mon_quantile(da_data, mhw_threshold=90.):
    """
    This function is designed for calculating the nmme 
    marine heat wave threshold.
    
    The threshold is calculated using a 3 month window
    to identified the X quantile value for each month.

    Parameters
    ----------
    da_data : xr.DataArray 
        For marine heat wave calculation, it should be the sst DataArray. 
        Since this function is designed for NMME model. The DataArray need 
        to be in the format of NMME model output.
    mhw_threshold : float
        the X quantile one wants to calculate. 

    Returns
    -------
    da_data_quantile : xr.DataArray
        The xr.DataArray for the quantile 

    Raises
    ------

    """
    
    da_data_quantile = xr.DataArray(coords={'X':da_data.X,
                                            'Y':da_data.Y,
                                            'month':np.arange(1,13),
                                            'L':da_data.L},
                                    dims = ['month','L','Y','X'])

    for i in range(1,13):
        if i == 1:
            mon_range = [12,1,2]
        elif i == 12 :
            mon_range = [11,12,1]
        else:
            mon_range = [i-1,i,i+1]

        da_data_quantile[i-1,:,:,:] = (da_data
                                 .where((da_data['S.month'] == mon_range[0])|
                                        (da_data['S.month'] == mon_range[1])|
                                        (da_data['S.month'] == mon_range[2]),drop=True)
                                 .stack(allens=('S','M'))
                                 .quantile(mhw_threshold*0.01, dim = 'allens', method='linear',skipna = True))

    return da_data_quantile
    
def nmme_mmm_3mon_quantile(da_data, mhw_threshold=90.):
    """
    This function is designed for calculating the nmme 
    marine heat wave threshold based on NMME MMM.
    
    The threshold is calculated using a 3 month window
    to identified the X quantile value for each month.

    Parameters
    ----------
    da_data : xr.DataArray 
        For marine heat wave calculation, it should be the sst DataArray. 
        Since this function is designed for NMME model. The DataArray need 
        to be in the format of NMME model output.
    mhw_threshold : float
        the X quantile one wants to calculate. 

    Returns
    -------
    da_data_quantile : xr.DataArray
        The xr.DataArray for the quantile 

    Raises
    ------

    """
    
    da_data_quantile = xr.DataArray(coords={'X':da_data.X,
                                            'Y':da_data.Y,
                                            'month':np.arange(1,13),
                                            'L':da_data.L},
                                    dims = ['month','L','Y','X'])

    for i in range(1,13):
        if i == 1:
            mon_range = [12,1,2]
        elif i == 12 :
            mon_range = [11,12,1]
        else:
            mon_range = [i-1,i,i+1]

        da_data_quantile[i-1,:,:,:] = (da_data
                                 .where((da_data['S.month'] == mon_range[0])|
                                        (da_data['S.month'] == mon_range[1])|
                                        (da_data['S.month'] == mon_range[2]),drop=True)
                                 .stack(allens=('S'))
                                 .quantile(mhw_threshold*0.01, dim = 'allens', method='linear',skipna = True))

    return da_data_quantile

            