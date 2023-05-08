import xarray as xr
import numpy as np
import warnings
import json
import os
from . import utils


def MHW_metrics_cmip(ds, baseline_years, baseline_type, var="tos"):

    """
    Calculate marine heatwave (MHW) metrics for a CMIP (Coupled Model Intercomparison Project) dataset, 
    including the number of MHW events, marine heat spike (MHS) events, mean anomaly, and cumulative anomaly.

    :param ds: An xarray Dataset containing sea surface temperature (SST) data.
    :type ds: xarray.Dataset

    :param baseline_years: The number of years to use for calculating the baseline climatology.
    :type baseline_years: int

    :param baseline_type: The type of baseline to use for calculating MHW events, either "fixed_baseline" or "moving_baseline".
    :type baseline_type: str

    :param var: The variable name for sea surface temperature in the input dataset.
    :type var: str, optional, default: "tos"

    :return: An xarray Dataset containing the MHW metrics for each year, including MHS events, MHW events, mean anomaly,
             and cumulative anomaly.
    :rtype: xarray.Dataset
    """

    year_length = 365
    if ds.time.dtype == "datetime64[ns]":
        # If the dataset has leapyears create a boolean mask to identify leap year February 29th
        leap_year_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
        # Remove leap year February 29th from the dataset
        ds = ds.where(~leap_year_feb29, drop=True)

    else:
        time_index = ds.indexes["time"]
        if hasattr(time_index, "calendar"):
            calendar = time_index.calendar
            if calendar == "360_day":
                year_length = 360

    lat, lon, lat_idx, lon_idx = utils.get_1d_lat_lon(ds)

    y_i = ds.time.dt.year.min().item()
    y_f = ds.time.dt.year.max().item()

    if baseline_type == "fixed_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(y_i, y_i + baseline_years)))
        thresholds = utils.compute_thresholds(
            baseline, year_length=year_length, var=var, lat=lat_idx, lon=lon_idx
        )

    # Create simple lat and lon coordinates
    # Initialize the output dataset
    ds_out = xr.Dataset(
        {
            "MHS": (
                ["time", "lat", "lon"],
                np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
            ),
            "MHW": (
                ["time", "lat", "lon"],
                np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
            ),
            "mean_anomaly": (
                ["time", "lat", "lon"],
                np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
            ),
            "cumulative_anomaly": (
                ["time", "lat", "lon"],
                np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
            ),
        },
        coords={
            "time": np.arange(y_i + baseline_years, y_f + 1),
            "lat": lat,
            "lon": lon,
        },
    )

    # Group the input dataset by year
    grouped_years = ds.groupby("time.year")

    for year, group in grouped_years:
        if year <= y_i + baseline_years - 1:
            continue
        if ds.time.dtype == "datetime64[ns]":
            group = group.where(~(group.time.dt.dayofyear == 366), drop=True)

        print(year, end=" ")

        if baseline_type == "moving_baseline":
            baseline = ds.sel(
                time=ds.time.dt.year.isin(range(year - baseline_years, year))
            )
            thresholds = utils.compute_thresholds(
                baseline, lat=lat_idx, lon=lon_idx, year_length=year_length
            )

        # some code to compute the MHSs per gridcell on that year
        year_thresholds = thresholds.sel(dayofyear=group.time.dt.dayofyear)

        mhs = (group[var] > year_thresholds["pctl"]).where(
            year_thresholds["pctl"].notnull()
        )
        mhw = np.apply_along_axis(utils.mhs_to_mhw, 0, mhs)
        # Update the output dataset

        ds_out["MHS"].loc[{"time": year}] = np.sum(mhs, axis=0)
        ds_out["MHW"].loc[{"time": year}] = np.sum(mhw, axis=0)

        anomaly = (group[var] - year_thresholds["clim"]).values
        anomaly = np.where(mhw == 0, np.nan, anomaly)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_an = np.nanmean(anomaly, axis=0)
            cum_an = np.nansum(anomaly, axis=0)

        ds_out["mean_anomaly"].loc[{"time": year}] = mean_an
        ds_out["cumulative_anomaly"].loc[{"time": year}] = cum_an
        ds_out["cumulative_anomaly"] = ds_out["cumulative_anomaly"].where(
            ds_out["MHS"].notnull()
        )

    return ds_out

def MHW_metrics_satellite(
    ds,
    baseline_years,
    baseline_type,
    out_folder,
    var="analysed_sst",
    distribution=False,
    error=False,
):
    
    year_length = 365
    if ds.time.dtype == "datetime64[ns]":
        # If the dataset has leapyears create a boolean mask to identify leap year February 29th
        leap_year_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
        # Remove leap year February 29th from the dataset
        ds = ds.where(~leap_year_feb29, drop=True)

    # else:
    #     time_index = ds.indexes["time"]
    #     if hasattr(time_index, "calendar"):
    #         calendar = time_index.calendar
    #         if calendar == "360_day":
    #             year_length = 360

    y_i = ds.time.dt.year.min().item()
    y_f = ds.time.dt.year.max().item()

    if baseline_type == "fixed_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(y_i, y_i + baseline_years)))
        thresholds = utils.compute_thresholds(baseline, year_length=year_length, var=var)

    lat = ds.lat
    lon = ds.lon

    # Initialize the output dataset
    data_vars = {}
    metrics = [
        "MHS",
        "MHW",
        "MHW_cat_2",
        "MHW_cat_3",
        "MHW_cat_4",
        "mean_anomaly",
        "cumulative_anomaly",
        "mean_duration",
    ]
    for metric in metrics:
        data_vars[metric] = (
            ["time", "lat", "lon"],
            np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
        )
        ds_out = xr.Dataset(
            data_vars,
            coords={
                "time": np.arange(y_i + baseline_years, y_f + 1),
                "lat": lat,
                "lon": lon,
            },
        )

    if error != False:
        suffixes = ["pos", "neg"]
        for metric in metrics:
            if metric not in ["MHW_cat_2", "MHW_cat_3", "MHW_cat_4"]:
                for app in suffixes:
                    ds_out[f"{metric}_{app}"] = xr.DataArray(
                        np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
                        dims=["time", "lat", "lon"],
                        coords={
                            "time": np.arange(y_i + baseline_years, y_f + 1),
                            "lat": lat,
                            "lon": lon,
                        },
                    )
    grouped_years = ds.groupby("time.year")

    distribution_metrics = [
        "MHS_days_year",
        "MHW_days_year",
        "MHW_cat_2_days_year",
        "MHW_cat_3_days_year",
        "MHW_cat_4_days_year",
        "MHW_event_duration",
        "MHW_anual_cumulative_anomaly",
        "MHW_event_mean_anomaly",
    ]

    ofo = out_folder + f"{baseline_type}_{baseline_years}_year/"
    if not os.path.exists(ofo):
        os.makedirs(ofo)

    for year, group in grouped_years:
        if year <= y_i + baseline_years - 1:
            continue
        if ds.time.dtype == "datetime64[ns]":
            group = group.where(~(group.time.dt.dayofyear == 366), drop=True)

        distributions = {key: [] for key in distribution_metrics}

        print(year, end=", ")

        if os.path.exists(f"{ofo}/MHW_{year}.nc"):
            continue

        if baseline_type == "moving_baseline":
            baseline = ds.sel(
                time=ds.time.dt.year.isin(range(year - baseline_years, year))
            )
            thresholds = utils.compute_thresholds(baseline, year_length=year_length, var=var)
        # some code to compute the MHSs per gridcell on that year
        year_thresholds = thresholds.sel(dayofyear=group.time.dt.dayofyear)

        sst = group[var]

        def get_metrics(
            sst, year_thresholds, year, ds_out, distribution=distribution, app=""
        ):
            mhs = (
                (sst > year_thresholds["pctl"])
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            mhw = np.apply_along_axis(utils.mhs_to_mhw, 0, mhs)
            ds_out[f"MHS{app}"].loc[{"time": year}] = np.sum(mhs, axis=0)
            ds_out[f"MHW{app}"].loc[{"time": year}] = np.sum(mhw, axis=0)

            dif = year_thresholds["pctl"] - year_thresholds["clim"]

            # Computing MHW categories

            mhw_cat_2 = (
                (sst > (year_thresholds["pctl"] + dif))
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            ds_out[f"MHW_cat_2"].loc[{"time": year}] = np.sum(mhw_cat_2, axis=0)

            mhw_cat_3 = (
                (sst > (year_thresholds["pctl"] + 2 * dif))
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            ds_out[f"MHW_cat_3"].loc[{"time": year}] = np.sum(mhw_cat_3, axis=0)

            mhw_cat_4 = (
                (sst > (year_thresholds["pctl"] + 3 * dif))
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            ds_out[f"MHW_cat_4"].loc[{"time": year}] = np.sum(mhw_cat_4, axis=0)

            # Computing anomalies

            anomaly = (sst - year_thresholds["clim"]).values
            anomaly = np.where(mhw == 0, np.nan, anomaly)

            durs = utils.find_mhw_durations(mhw)
            mean_an_event = utils.find_mean_anomaly(anomaly)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_anomaly = np.nanmean(anomaly, axis=0)
                cumulative_anomaly = np.nansum(anomaly, axis=0)
                mean_duration = np.nanmean(durs, axis=0)

            ds_out[f"mean_anomaly{app}"].loc[{"time": year}] = mean_anomaly
            ds_out[f"cumulative_anomaly{app}"].loc[{"time": year}] = cumulative_anomaly
            ds_out[f"cumulative_anomaly{app}"] = ds_out[
                f"cumulative_anomaly{app}"
            ].where(ds_out[f"MHS{app}"].notnull())
            ds_out[f"mean_duration{app}"].loc[{"time": year}] = mean_duration
            ds_out[f"mean_duration{app}"] = ds_out[f"mean_duration{app}"].where(
                ds_out[f"MHS{app}"].notnull()
            )

            ds_out_year = ds_out.where(ds_out.time == year, drop=True)
            ds_out_year.to_netcdf(f"{ofo}/MHW_{year}.nc")

            if distribution != False:
                distributions["MHW_event_duration"].append(utils.nonzero_and_not_nan(durs))
                distributions["MHW_event_mean_anomaly"].append(
                    utils.nonzero_and_not_nan(mean_an_event)
                )
                distributions["MHS_days_year"].append(
                    utils.nonzero_and_not_nan(ds_out["MHS"].values)
                )
                distributions["MHW_days_year"].append(
                    utils.nonzero_and_not_nan(ds_out["MHW"].values)
                )
                distributions["MHW_cat_2_days_year"].append(
                    utils.nonzero_and_not_nan(ds_out["MHW_cat_2"].values)
                )
                distributions["MHW_cat_3_days_year"].append(
                    utils.nonzero_and_not_nan(ds_out["MHW_cat_3"].values)
                )
                distributions["MHW_cat_4_days_year"].append(
                    utils.nonzero_and_not_nan(ds_out["MHW_cat_4"].values)
                )
                distributions["MHW_anual_cumulative_anomaly"].append(
                    utils.nonzero_and_not_nan(ds_out["cumulative_anomaly"].values)
                )

                histograms = {}
                bin_params = {
                    "MHS_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_cat_2_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_cat_3_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_cat_4_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_event_mean_anomaly": {"range": (0, 7.01), "bin_width": 0.01},
                    "MHW_anual_cumulative_anomaly": {
                        "range": (0, 1001),
                        "bin_width": 1,
                    },
                    "MHW_event_duration": {"range": (0, 366), "bin_width": 1},
                }

                for metric in [
                    "MHS_days_year",
                    "MHW_days_year",
                    "MHW_cat_2_days_year",
                    "MHW_cat_3_days_year",
                    "MHW_cat_4_days_year",
                    "MHW_event_duration",
                    "MHW_anual_cumulative_anomaly",
                    "MHW_event_mean_anomaly",
                ]:
                    histograms[metric] = {}
                    distributions[metric] = utils.flatten(distributions[metric])
                    bin_edges = np.arange(
                        bin_params[metric]["range"][0],
                        bin_params[metric]["range"][1],
                        bin_params[metric]["bin_width"],
                    )
                    hist, bin_edges = np.histogram(
                        distributions[metric], bins=bin_edges
                    )
                    histograms[metric]["hist"] = [float(i) for i in hist]
                    histograms[metric]["bin_edges"] = [float(i) for i in bin_edges]

                fold_distr = ofo + "distributions/"
                if not os.path.exists(fold_distr):
                    os.makedirs(fold_distr)
                file_distr = fold_distr + f"distr_{year}.json"
                with open(file_distr, "w") as outfile:
                    json.dump(histograms, outfile)

        get_metrics(sst, year_thresholds, year, ds_out, distribution=distribution)

        if error != False:
            sst_pos = sst + group["analysis_error"]
            sst_neg = sst - group["analysis_error"]
            get_metrics(
                sst_pos, year_thresholds, year, ds_out, distribution=False, app="_pos"
            )
            get_metrics(
                sst_neg, year_thresholds, year, ds_out, distribution=False, app="_neg"
            )

def MHW_metrics_one_point(
    ds,
    baseline_years,
    baseline_type,
    year,
    var="analysed_sst",
    error=False,
    window=11,
    q=90,
):
    year_length = 365
    if ds.time.dtype == "datetime64[ns]":
        # If the dataset has leapyears create a boolean mask to identify leap year February 29th
        leap_year_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
        # Remove leap year February 29th from the dataset
        ds = ds.where(~leap_year_feb29, drop=True)

    y_i = ds.time.dt.year.min().item()
    y_f = ds.time.dt.year.max().item()

    def get_threshold(baseline):
        thres = utils.get_win_pctl(
            baseline[var].values, window, q, year_length=year_length
        )
        clim = utils.get_win_clim(baseline[var].values, window, year_length=year_length)
        baseline = baseline.sortby("time")
        ds_out = baseline.isel(time=slice(0, year_length))
        ds_out = ds_out.groupby("time.dayofyear").mean(dim="time")
        ds_out = ds_out.drop_vars(var)
        if error == True:
            ds_out = ds_out.drop_vars("analysis_error")
        ds_out["pctl"] = (("dayofyear"), thres)
        ds_out["clim"] = (("dayofyear"), clim)
        ds_out = ds_out.where(ds_out.pctl < 9999)
        return ds_out

    if baseline_type == "fixed_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(y_i, y_i + baseline_years)))
        ds_out = get_threshold(baseline)

    if baseline_type == "moving_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(year - baseline_years, year)))
        ds_out = get_threshold(baseline)

    ds_y = ds.sel(time=ds.time.dt.year.isin([year]))
    ds_y = ds_y.groupby("time.dayofyear").mean(dim="time")
    # ds_y = ds_y.where(~(ds_y.dayofyear == 366), drop=True)

    ds_out["sst"] = (("dayofyear"), ds_y[var].values)
    if error == True:
        ds_out["error"] = (("dayofyear"), ds_y["analysis_error"].values)

        for i, typ in enumerate(['pos', 'neg']):

            sign = (-1)**(i)
            sst = ds_out["sst"].values + sign * ds_out["error"].values
            mhs = (
                (sst > ds_out["pctl"]).where(ds_out["pctl"].notnull()).values
            )
            mhw = utils.mhs_to_mhw(mhs)
            anomaly = (sst - ds_out["clim"]).values

            ds_out['MHS_' + typ] = (('dayofyear'), mhs)
            ds_out['MHW_' + typ] = (('dayofyear'), mhw)
            ds_out['anomaly_' + typ] = (('dayofyear'), anomaly)


    mhs = (ds_out["sst"] > ds_out["pctl"]).where(ds_out["pctl"].notnull()).values
    mhw = utils.mhs_to_mhw(mhs)
    anomaly = (sst - ds_out["clim"]).values
    anomaly = np.where(mhw == 0, np.nan, anomaly)

    ds_out["MHS"] = (("dayofyear"), mhs)
    ds_out["MHW"] = (("dayofyear"), mhw)
    ds_out["anomaly"] = (("dayofyear"), anomaly)

    return ds_out
















