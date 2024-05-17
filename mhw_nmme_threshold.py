import numpy as np
import xarray as xr


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

            