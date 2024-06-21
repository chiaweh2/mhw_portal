# %%
"""
# NMME Temperature Observations to Avoid Loggerheads (TOTAL)
 
The script calculate the TOTAL value from the start of the nmme 1991 based on 
Brodie et al., [2023]

Using NMME model hindcast and forecast obtained from 
http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/
with variable SST

"""


# %%
# start a local cluster
import json
import subprocess
from datetime import date
import warnings
import numpy as np
import xarray as xr
from dask.distributed import Client
from nmme_download import iri_nmme_models
from nmme_hci_climo_threshold import read_marine_index_mask
from nmme_monthly_mhw import read_nmme_onlist
from nmme_hci_monthly import output_format
import time





warnings.filterwarnings("ignore")

### open local cluster
client = Client(processes=False)
print(client)
print(client.cluster.dashboard_link)

###### Setting ######
# specify date
today = date.today()
# setup the new output file name (same as date of download)
dateform = today.strftime("%y_%m_%d")

# directory where new simulation (inputs) and mhw forecast (outputs) is located
BASEDIR = '/Datasets.private/marinehw/nmme_sst_raw/'

# directory where sst threshold/climatology/trend (inputs) is located
PREDIR = '/Datasets.private/marinehw/nmme_sst_stat/'

# directory where nmme mhw probability is located
OUTDIR = '/Datasets.private/marinehw/nmme_mhw_prob/'

# directory where the marine index mask is located
MASKDIR = '/Datasets.private/marinehw/nmme_marine_index_mask/'

# output filename date, MHW prediction generated date
date = dateform

# used model list
with open('model_use_list.json','r',encoding='utf-8') as f:
    json_dict = json.load(f)
model_use_list = json_dict['model_use_list']

dict_model = iri_nmme_models()
avai_model_list = list(dict_model.keys())

################################## Main program start #####################################
start_time = time.time()
dict_da = read_nmme_onlist(model_use_list,avai_model_list,BASEDIR,PREDIR,start_year=1991,lazy=False,chunks = {'M':-1,'L':-1,'S':-1})
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time/60.)
# two times faster when  single chunk is assigned on the S dim then just following the original 1 S 1 chunk
