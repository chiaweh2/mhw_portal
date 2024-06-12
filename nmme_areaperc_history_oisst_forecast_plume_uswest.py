"""
The script is for creating the 
1. Historical time series combine with the forecast MMM plot
2. limited historical time series with the forecast plume plot

for the EEZ coastal region

"""
import warnings
from datetime import date
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client
import nmme_areaperc_history_oisst_forecast_plume as hf
import nmme_areaperc_forecast_uswest as future

warnings.filterwarnings("ignore")

##################################     Main script     ######################################
if __name__ == "__main__":
    ### open local cluster
    client = Client(processes=False)
    region_limit = [-160,-115,25,60]
    PREMONTH = 24                          # historical record to include in plume plot (month)
    OUTPUTDIR = '/home/chsu/mhw_portal/figures/'
    # OUTPUTDIR = '/httpd-test/psd/marine-heatwaves/img/'

    #### Calculating area percentage for observational data
    obs_mask_dict = hf.all_obs_area_mask()
    da_mask_01 = obs_mask_dict['mhw_trend']
    da_mask_01_detrend = obs_mask_dict['mhw_detrended']
    da_unmask_01 = obs_mask_dict['global']

    # create observation eez mask
    x_list_mj,y_list_mj = future.read_eez()
    da_obs_region = future.region_mask(x_list_mj,y_list_mj,da_unmask_01,xname='lon',yname='lat')
    # if da_mask_01['X'].min().data<0:
    #     # change mask coordinate to -180-180
    #     da_obs_region['X'] = da_obs_region['X'].where(
    #         da_obs_region['X']<=180.,
    #         other=da_obs_region['X']-360.
    #     )

    # mhw area
    da_mask_ts = hf.area_weighted_sum(da_mask_01*da_obs_region, xname='lon', yname='lat')
    da_mask_ts_detrend = hf.area_weighted_sum(
        da_mask_01_detrend*da_obs_region,
        xname='lon',
        yname='lat'
        )

    # global ocean area
    da_unmask_ts = hf.area_weighted_sum(da_unmask_01*da_obs_region, xname='lon', yname='lat')
    da_unmask_ts = da_unmask_ts.drop_vars('month')

    # mhw area percentage
    da_perc_obs_ts = da_mask_ts/da_unmask_ts
    da_perc_obs_ts_detrend = da_mask_ts_detrend/da_unmask_ts


    #### Calculating area percentage for NMME with trend/detrended
    nmme_mask_dict = hf.nmme_newest_forecast(threshold=90)
    da_mask_01 = nmme_mask_dict['mhw_trend']
    da_unmask_01 = nmme_mask_dict['global']

    # create nmme eez mask
    da_nmme_region = future.region_mask(x_list_mj,y_list_mj,da_unmask_01,xname='X',yname='Y')
    # if da_mask_01['X'].min().data<0:
    #     # change mask coordinate to -180-180
    #     da_nmme_region['X'] = da_nmme_region['X'].where(
    #         da_nmme_region['X']<=180.,
    #         other=da_nmme_region['X']-360.
    #     )    

    # mhw area
    da_mask_ts = hf.area_weighted_sum(da_mask_01*da_nmme_region, xname='X', yname='Y')
    da_mask_ts = da_mask_ts*nmme_mask_dict['ens_mask']

    # global ocean area
    da_unmask_ts = hf.area_weighted_sum(da_unmask_01*da_nmme_region, xname='X', yname='Y')

    # store area percentage
    ds_perc = xr.Dataset()
    ds_perc['global'] = (da_mask_ts/da_unmask_ts).compute()

    # mhw area detrended
    da_mask_01 = nmme_mask_dict['mhw_detrended']
    da_mask_ts = hf.area_weighted_sum(da_mask_01*da_nmme_region, xname='X', yname='Y')
    da_mask_ts = da_mask_ts*nmme_mask_dict['ens_mask']

    ds_perc['global_detrend'] = (da_mask_ts/da_unmask_ts).compute()

    # create valid-time (leadtime) time stamp
    ini_year = ds_perc.S.dt.year.data
    ini_month = ds_perc.S.dt.month.data
    valid_time = xr.cftime_range(
        start=f'{ini_year}-{ini_month:02d}-01',
        periods=13,
        freq='MS'
        )[:-1]
    ds_perc['valid_time'] = xr.DataArray(valid_time,coords={'L':ds_perc['L']},dims={'L'})
    ds_perc = ds_perc.set_coords('valid_time')

    # ranking
    forecast_ranking = hf.find_rank(
        np.append(
            da_perc_obs_ts.data,
            ds_perc['global'].mean(dim='M').mean(dim='model')[0].data
        )
    )
    forecast_detrend_ranking = hf.find_rank(
        np.append(
            da_perc_obs_ts_detrend.data,
            ds_perc['global_detrend'].mean(dim='M').mean(dim='model')[0].data
        )
    )
    ranking = hf.find_rank(da_perc_obs_ts.data)
    detrend_ranking = hf.find_rank(da_perc_obs_ts_detrend.data)

    #### plot with trend plume plot
    fig2, ax2 = hf.plot_plume(ds_perc['global'],da_perc_obs_ts,PREVMONTH=PREMONTH,RANK=False)
    ax2.set_title("EEZ Ocean MHW Area", color='black', weight='bold',size=15,pad=20)
    ax2.set_yticks(np.arange(0,1.01,0.1))
    ax2.set_yticklabels([f"{int(n*100)}%" for n in np.arange(0,1.01,0.1)])
    new_date = (
        date.fromisoformat(f'{ds_perc.valid_time[0].dt.year.data}-'+
                           f'{ds_perc.valid_time[0].dt.month.data:02d}-'+
                           f'{ds_perc.valid_time[0].dt.day.data:02d}')
    )
    ax2.text(1.02, 0.1,
            f'{new_date.strftime("%b %Y")} forecast',
            color='indigo',
            size=13,
            transform=ax2.transAxes)
    ax2.text(1.02, 0.1-0.06,
            f'ranks {forecast_ranking} of {len(da_perc_obs_ts.time)+1} months',
            color='indigo',
            size=13,
            transform=ax2.transAxes)
    fig2.savefig(f'{OUTPUTDIR}MHW_area_plume_eez.png',
            dpi=300,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            transparent=False,
            bbox_inches="tight",
            pad_inches=None)
    plt.clf()

    fig3, ax3 = hf.plot_plume(ds_perc['global_detrend'],da_perc_obs_ts_detrend,PREVMONTH=PREMONTH,RANK=False)
    ax3.set_yticks(np.arange(0,1.01,0.1))
    ax3.set_yticklabels([f"{int(n*100)}%" for n in np.arange(0,1.01,0.1)])
    ax3.set_title(
        "EEZ Ocean MHW Area (without trend)",
        color='black',
        weight='bold',
        size=15,
        pad=20
    )
    ax3.text(1.02, 0.1,
            f'{new_date.strftime("%b %Y")} forecast',
            color='indigo',
            size=13,
            transform=ax3.transAxes)
    ax3.text(1.02, 0.1-0.06,
            f'ranks {forecast_detrend_ranking} of {len(da_perc_obs_ts_detrend.time)+1} months',
            color='indigo',
            size=13,
            transform=ax3.transAxes)
    fig3.savefig(f'{OUTPUTDIR}MHW_area_plume_detrend_eez.png',
            dpi=300,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            transparent=False,
            bbox_inches="tight",
            pad_inches=None)
    plt.clf()

    # fig4, ax4 = plot_history_forecast(ds_perc,da_perc_obs_ts,da_perc_obs_ts_detrend)
    # fig4.savefig('/home/chsu/MHW/figure/MHW_area_history_forecast.png',
    #         dpi=300,
    #         facecolor='w',
    #         edgecolor='w',
    #         orientation='portrait',
    #         transparent=False,
    #         bbox_inches="tight",
    #         pad_inches=None)
    # plt.clf()

    fig5, ax5 = hf.plot_history(da_perc_obs_ts,da_perc_obs_ts_detrend,RANK=False,LEGEND=True)
    # legend = ax5.legend(loc='upper right',fontsize=14,frameon=False)
    # legend.set_bbox_to_anchor((1.21, 0.2))
    ax5.set_title(
        "EEZ Ocean MHW area (OISSTv2)",
        color='black',
        weight='bold',
        size=20,
        pad=20
    )
    ax5.set_yticks(np.arange(0,1.01,0.1))
    ax5.set_yticklabels([f"{int(n*100)}%" for n in np.arange(0,1.01,0.1)])
    # ranking text outside the plot
    new_date = (
        date.fromisoformat(
            f'{da_perc_obs_ts.time[-1].dt.year.data}-'+
            f'{da_perc_obs_ts.time[-1].dt.month.data:02d}-'+
            f'{da_perc_obs_ts.time[-1].dt.day.data:02d}'
        )
    )
    ax5.text(1.02, 0.95,
            f'{new_date.strftime("%b %Y")} ',
            color='C1',
            size=13,
            transform=ax5.transAxes)
    ax5.text(1.02, 0.95-0.05,
            f'ranks {ranking} of {len(da_perc_obs_ts.time)} months',
            color='C1',
            size=13,
            transform=ax5.transAxes)
    ax5.text(1.02, 0.95-0.1,
            f'ranks {detrend_ranking} of {len(da_perc_obs_ts_detrend.time)} months',
            color='C0',
            size=13,
            transform=ax5.transAxes)
    fig5.savefig(f'{OUTPUTDIR}MHW_area_history_only_eez.png',
            dpi=300,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            transparent=False,
            bbox_inches="tight",
            pad_inches=None)
    plt.clf()

    client.close()
