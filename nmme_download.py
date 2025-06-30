"""
This script is to downloaded NMME model 
from IRI to the new unified folder. 
/Datasets.private/marinehw/nmme_sst_raw
The unified folder is the place to share all used 
NMME model raw output.

The script is designed to check existing on prem file
before downloading. This should be 

!!!!
If there is overlapping time period in hindcast and forecast
the hindcast one is preserved
!!!!
"""

#%%
import glob
import datetime
import cftime
import xarray as xr
import json
import os



def iri_nmme_models(json_path='iri_nmme_models.json'):
    """
    Load the NMME models locations for both hindcast and forecast output from a JSON file.
    Returns only the model_iri_locations dictionary.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file {json_path} not found.")

    with open(json_path, 'r') as f:
        data = json.load(f)

    model_iri_locations = data.get("model_iri_locations")
    return model_iri_locations


if __name__=="__main__":

    # user settings
    START_YEAR = 1991
    OUTPUTDIR = '/Datasets.private/marinehw/nmme_sst_raw/'

    # date form (download stamp)
    today = datetime.date.today()
    dateform = today.strftime("%y_%m_%d")

    # get all hindcast and forecast IRI OPeNDAP URL
    NMME_IRI_LOC = iri_nmme_models()

    # opendap lazy loading (IRI availibility)
    dict_model = {}
    nmme_name = []
    for name,links in NMME_IRI_LOC.items():
        print('=============')
        print(f'downloading {name}')
        nmme_name.append(name)
        dict_model[name] = []

        for link in links:
            # open links to the whole set of available
            ds_temp = xr.open_dataset(link, chunks={'M':1,'L':1,'S':1},decode_times=False)
            dict_model[name].append(ds_temp)

            # all available initial time in the link
            initial_time = cftime.num2date(
                ds_temp.S.values,
                ds_temp.S.units,
                calendar='360_day'
            )

            # download each initial time seperately
            for s,S in enumerate(ds_temp.S.data):
                if initial_time[s].year >= START_YEAR:
                    FILENAME = f'{name}_forecast_{dateform}_{initial_time[s].year:04d}{initial_time[s].month:02d}.nc'

                    # find on prem storage availability
                    FILENAME_WILD = f'{name}_forecast_??_??_??_{initial_time[s].year:04d}{initial_time[s].month:02d}.nc'
                    on_disc_list = glob.glob(
                        f'{OUTPUTDIR}'+FILENAME_WILD
                    )
                    if len(on_disc_list) > 0:
                        print('forecast file already exist on prem')
                        print(on_disc_list)
                    else:
                        print(f'downloading forecast file {FILENAME}')
                        print(f'from {link}')
                        ds_temp.sel(S=S).to_netcdf(
                            f'{OUTPUTDIR}'+ FILENAME
                        )
