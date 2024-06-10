import json
import warnings
from datetime import date
import xarray as xr
import numpy as np
from dask.distributed import Client
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapefile as shp
from nmme_download import iri_nmme_models
from nmme_monthly_mhw import read_nmme_onlist



def deg_to_rad(degree):
    return degree*np.pi/180.

def rad_to_deg(radian):
    return radian*180./np.pi

def distance_along_lat(delta_longitude,latitude):
    earth_radius = 6371.  # km
    delta_distance = (
        earth_radius
        *np.cos(deg_to_rad(latitude))
        *deg_to_rad(delta_longitude)
    )
    return delta_distance

def delta_lon_along_lat(delta_distance,latitude):
    earth_radius = 6371.  # km
    delta_longitude_rad = (
        delta_distance/earth_radius/np.cos(deg_to_rad(latitude))
    )

    return rad_to_deg(delta_longitude_rad)

def polygon_off_coast(distance,lon_list,lat_list,eastward_ext=False,lon_lim = [-130,-120],lat_lim = [35,40]):
    if eastward_ext:
        lon_list_extend = lon_list + delta_lon_along_lat(distance,lat_list)
    else: 
        lon_list_extend = lon_list - delta_lon_along_lat(distance,lat_list)

    # lon_list_poly = np.concatenate([lon_list,lon_list_extend[::-1],np.array([lon_list[0]])])
    # lat_list_poly = np.concatenate([lat_list,lat_list[::-1],np.array([lat_list[0]])])

    # point orientation: top right point, x away from coastline,  bottom right, top right
    lon_list_poly = np.concatenate([np.array([lon_lim[1]]),lon_list_extend[::-1],np.array([lon_lim[1]]),np.array([lon_lim[1]])])
    lat_list_poly = np.concatenate([np.array([lat_list[-1]]),lat_list[::-1],np.array([lat_lim[0]]),np.array([lat_list[-1]])])

    indi = np.where(lon_list_poly<0)
    lon_list_poly[indi] = lon_list_poly[indi]+360.

    poly_list = []
    for i, lon in enumerate(lon_list_poly):
        poly_list.append((lon,lat_list_poly[i]))
    poly = Polygon(poly_list)

    return {
        'lon':lon_list_poly,
        'lat':lat_list_poly,
        'polygon':poly
    }

def polygon_inland(distance,lon_list,lat_list,eastward_ext=False,outpoint = [-120,31]):
    if eastward_ext:
        lon_list_extend = lon_list + delta_lon_along_lat(distance,lat_list)
    else: 
        lon_list_extend = lon_list - delta_lon_along_lat(distance,lat_list)

    # point orientation:  x inland from coastline,top left, bottoom left, bottom right
    lon_list_poly = np.concatenate([lon_list_extend,np.array([outpoint[0]]),np.array([outpoint[0]]),np.array([lon_list_extend[0]])])
    lat_list_poly = np.concatenate([lat_list,np.array([lat_list[-1]]),np.array([outpoint[1]]),np.array([lat_list[0]])])

    indi = np.where(lon_list_poly<0)
    lon_list_poly[indi] = lon_list_poly[indi]+360.

    poly_list = []
    for i, lon in enumerate(lon_list_poly):
        poly_list.append((lon,lat_list_poly[i]))
    poly = Polygon(poly_list)

    return {
        'lon':lon_list_poly,
        'lat':lat_list_poly,
        'polygon':poly
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    ### open local cluster
    client = Client(n_workers=2,threads_per_worker=60,processes=False)

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

    # output filename date, MHW prediction generated date
    date = dateform

    # MHW threshold for prediction
    threshold = [90]

    # used model list
    with open('model_use_list.json','r',encoding='utf-8') as f:
        json_dict = json.load(f)
    model_use_list = json_dict['model_use_list']

    dict_model = iri_nmme_models()
    avai_model_list = list(dict_model.keys())

    ################################## Main program start #####################################

    dict_da = read_nmme_onlist(model_use_list,avai_model_list,BASEDIR,PREDIR)
    da_mask = dict_da['da_allmodel_mask'].isel(S=-1,L=0)   # assuming all initialization has the same grid
    da_mask = da_mask.where(
        (da_mask.X>-127+360)&
        (da_mask.X<-115+360)&
        (da_mask.Y>30)&
        (da_mask.Y<41),
        drop=True
    )

    res = 'l'
    sf = shp.Reader(f'/home/chsu/data/gshhg-shp-2/GSHHS_shp/{res}/GSHHS_{res}_L1')

    ###### HCI mask ######
    x_list = []
    y_list = []
    x_lim = [-130,-120]
    y_lim = [35,40]
    for nf,feature in enumerate(sf.shapeRecords()):
        x=[]
        y=[]
        for i in feature.shape.points[:]:
            if i[0]>=x_lim[0] and i[0]<=x_lim[-1] and i[1]>=y_lim[0] and i[1]<=y_lim[-1]:
                if nf == 3:
                    x.append(i[0])
                    y.append(i[1])
        x_list += x
        y_list += y

    x_list = np.array(x_list)
    y_list = np.array(y_list)

    # remove SF bay and san pablo bay
    x_list_new = np.concatenate([x_list[0:15],x_list[27:]])
    y_list_new = np.concatenate([y_list[0:15],y_list[27:]])

    dict_poly_150 = polygon_off_coast(150.,x_list_new,y_list_new,eastward_ext=False)
    dict_poly_75 = polygon_off_coast(75.,x_list_new,y_list_new,eastward_ext=False)

    ###### TOTAL mask ######
    x_list = []
    y_list = []
    x_lim = [-120.45,-115]
    y_lim = [30.9,36]
    outpoint = [-120,31]
    for nf,feature in enumerate(sf.shapeRecords()):
        x=[]
        y=[]
        for i in feature.shape.points[:]:
            if i[0]>=x_lim[0] and i[0]<=x_lim[-1] and i[1]>=y_lim[0] and i[1]<=y_lim[-1]:
                if nf == 3:
                    x.append(i[0])
                    y.append(i[1])
        x_list += x
        y_list += y

    x_list = np.array(x_list)
    y_list = np.array(y_list)

    dict_poly_inland = polygon_inland(100.,x_list,y_list,eastward_ext=True,outpoint = outpoint)


    ### create masks based on data mask and created polygon
    lim_lon_array = da_mask.X.data
    lim_lat_array = da_mask.Y.data
    polygons = [
        dict_poly_150['polygon'],
        dict_poly_75['polygon'],
        dict_poly_inland['polygon']
    ]
    da_masks = []

    for polygon in polygons:
        da_mask_region = da_mask.copy()*np.nan
        da_mask_region = da_mask_region.drop_vars(['S','L'])
        for lon in lim_lon_array:
            for lat in lim_lat_array:
                if da_mask.sel(X=lon,Y=lat).data == 1:
                    point = Point(lon, lat)
                    if polygon.contains(point):
                        ii,  = np.where(lim_lon_array==lon)
                        jj,  = np.where(lim_lat_array==lat)
                        da_mask_region[jj,ii] = 1
        da_masks.append(da_mask_region)
    
    ds_mask = xr.Dataset()
    ds_mask['HCI_150km'] = da_masks[0]
    ds_mask['HCI_75km'] = da_masks[1]
    ds_mask['TOTAL'] = da_masks[2]
    
    ds_mask.to_netcdf('/Datasets.private/marinehw/nmme_marine_index_mask/hci_total_mask.nc')