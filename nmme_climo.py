# %%
"""
# NMME Marine Heat Wave
 
The script calculate the climatology based on 
Mike Jacox et al., [2022]

Using NMME model hindcast and forecast obtained from 
http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
with variable SST

The steps are described below.
------------------------------
calculate monthly ensemble mean climatology for each model 
initialization month and lead time

"""


# %%
# start a local cluster
import sys
import warnings
import cftime
import xarray as xr
from dask.distributed import Client
from nmme_download import iri_nmme_models

warnings.simplefilter("ignore")


def nmme_ens_climo(ds,climo_dim='S',ens_dim='M'):
    """
    Calculate the climatology based on `climo_dim`
    """

    # calculate ensemble mean
    ds_ensmean = ds.mean(dim=ens_dim)

    # calculate monthly climatology of each initialization month and lead time
    ds_ensmean_climatology = ds_ensmean.groupby(f'{climo_dim}.month').mean(dim=climo_dim)

    return ds_ensmean_climatology

def read_nmme(
        forecast_files : str,
        model : str,
        base_dir : str = '/Datasets.private/marinehw/nmme_sst_raw/',
        chunks : dict = None
) -> xr.Dataset:
    """read the nmme data

    Parameters
    ----------
    forecast_files : str
        wildcard string describing the nmme forecat file
        in the format of {model}_forecast_??_??_??_??????.nc
        with the full path in front of the filename
    model : str
        model name used in the file name
    base_dir : str, optional
        the dir path to the raw nmme sst file,
        by default '/Datasets.private/marinehw/nmme_sst_raw/'
    chunks : dict, optional
        a dict object defining how `open_mfdataset()` method is 
        going to chunk your file. By default, None (means letting
        the method to used the data's original chunking)

    Returns
    -------
    xr.Dataset
        The nmme model with raw sst field
    """
    if chunks is None:
        chunks = {}

    print('reading data')
    ds_model = xr.open_mfdataset(
        forecast_files,
        decode_times=False,
        concat_dim='S',
        combine='nested',
        chunks=chunks
    )

    # make sure variable name is sst
    all_var = list(ds_model.variables)
    all_coords = list(ds_model.coords)
    for var in all_var:
        if var not in all_coords:
            varname = var

    if varname != 'sst':
        ds_model = ds_model.rename({varname:'sst'})
        print(f'variable name convert from {varname} to sst')

    ds_model = ds_model.drop_duplicates(dim='S')

    ds_model['S'] = cftime.num2date(ds_model['S'],
                                    ds_model.S.units,
                                    calendar='360_day')

    da_model = ds_model['sst']

    if model in ['CanCM4i', 'GEM-NEMO']:
        # the land part is zero for these two models in the earlier forecast (check 199101)
        da_model = da_model.where(da_model!=0)
        # the unit is in Kelvin
        da_model = da_model-273.15

        da_mask = da_model.isel(S=0,L=0,M=0)
        da_mask = da_mask.where(da_mask.isnull(),other=1)
        da_model = da_model*da_mask

        ds_model['sst'] = da_model

    if model in ['CanCM4i-IC3','CanESM5']:
        # the unit is in Kelvin
        da_model = da_model-273.15

        # the land part is with value not NaN for this model
        #  using old version model to mask land region
        da_mask = xr.open_mfdataset(
            f'{base_dir}CanCM4i_forecast_??_??_??_??????.nc',
            decode_times=False,
            concat_dim='S',
            combine='nested'
        )['sst'].isel(S=0,L=0,M=0) #earlier forecast has 0 over land (check 199101)
        da_mask = da_mask.where(da_mask!=0)
        da_mask = da_mask.where(da_mask.isnull(),other=1)
        da_model = da_model*da_mask

        ds_model['sst'] = da_model

    if model in ['GEM5-NEMO','GEM5.2-NEMO']:
        # the unit is in Kelvin
        da_model = da_model-273.15

        # the land part is with value not NaN for this model
        #  using old version model to mask land region
        da_mask = xr.open_mfdataset(
            f'{base_dir}GEM-NEMO_forecast_??_??_??_??????.nc',
            decode_times=False,
            concat_dim='S',
            combine='nested'
        )['sst'].isel(S=0,L=0,M=0) #earlier forecast has 0 over land (check 199101)
        da_mask = da_mask.where(da_mask!=0)
        da_mask = da_mask.where(da_mask.isnull(),other=1)
        da_model = da_model*da_mask

        ds_model['sst'] = da_model

    return ds_model

if __name__ == "__main__":

    client = Client(n_workers=1,threads_per_worker=60,processes=False)
    print(client)
    print(client.cluster.dashboard_link)

    BASEDIR = '/Datasets.private/marinehw/nmme_sst_raw/'
    OUTPUTDIR = '/Datasets.private/marinehw/nmme_sst_stat/'
    if len(sys.argv) < 2:
        print("Usage: python nmme_climo.py <model name>")

    dict_model = iri_nmme_models()
    avai_model_list = list(dict_model.keys())

    # read user input
    model_list = []
    for model_name in sys.argv[1:]:
        if model_name in avai_model_list:
            model_list.append(str(model_name))
            print(f'{model_name} exist, add to process list')
        else:
            print(f'{model_name} does not exist')
            print(f'Make sure pick model from {avai_model_list}')

        # construct model list
        forecast_nmme_files = f'{BASEDIR}{model_name}_forecast_??_??_??_??????.nc'
        output_file = f'{OUTPUTDIR}{model_name}_climo.nc'

        # consistent climatology period
        climatology_period = [1991,2020]

        print('-------------')
        print(model_name)
        print('-------------')
        ds_nmme = read_nmme(
            forecast_files = forecast_nmme_files,
            model = model_name,
            chunks = {'S':-1,'M':-1,'L':-1,'X':-1,'Y':-1}
        )

        ds_nmme = ds_nmme.where(
            (ds_nmme['S.year']>=climatology_period[0])&
            (ds_nmme['S.year']<=climatology_period[1]),
            drop=True
        )

        print('calculating climatology')
        ds_ensmean_climo = nmme_ens_climo(ds_nmme,climo_dim='S',ens_dim='M').compute()

        print('file output')
        ds_ensmean_climo.to_netcdf(output_file)
