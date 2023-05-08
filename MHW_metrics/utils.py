import numpy as np
import copy


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def get_distr(da, zeros=True, above=None):
    
    darr = np.array(da)
    values = darr[darr<1000]
        
    if zeros==True:
        values = values[values>0]

    if above!=None:
        values = values[values>above]
    
    return values

def run_avg_per(sst, w=11):
    """
    Calculate the periodic moving average of a given sequence.

    This function calculates the periodic moving average of a given input
    array `sst` using a sliding window of size `w`. The function also handles
    periodicity by extending the input array to the left and right, which
    allows for accurate calculations near the edges of the input data.

    Parameters
    ----------
    sst : numpy.ndarray or list
        The input sequence for which the periodic moving average will be calculated.
    w : int, optional, default=11
        The window size for the moving average.

    Returns
    -------
    sst : numpy.ndarray
        The calculated periodic moving average of the input sequence with the same
        shape as the input.
    """

    var = np.array(sst)
    hw = w // 2

    # Pad the input array with periodic boundary conditions
    var_padded = np.pad(var, pad_width=(hw, hw), mode="wrap")

    # Calculate the moving average using convolution
    avg = np.convolve(var_padded, np.ones(w), "valid") / w

    return avg

def get_win_clim(sst, w=11, year_length=365):
    """
    Calculate day-of-year w-day window rolling mean and smooth with
    a 31-day periodic moving average.

    Parameters
    ----------
    sst : numpy.ndarray or list
        The input sequence for which the rolling percentile will be calculated.
    w : int, optional, default=11
        The window size for the rolling mean.

    Returns
    -------
    fin_run : numpy.ndarray
        The calculated day-of-year rolling percentile smoothed with a 31-day
        periodic moving average.
    """

    if np.isnan(sst[0]):
        return np.array([10000.0] * year_length)

    n = len(sst) // year_length
    var = np.reshape(np.array(sst), (n, year_length))
    hw = w // 2

    # Extend the input array with periodic boundary conditions
    var_ext = np.pad(var, pad_width=((0, 0), (hw, hw)), mode="wrap")

    # Calculate the rolling percentile
    clim = [np.mean(var_ext[:, i : i + 2 * hw + 1]) for i in range(year_length)]

    fin = np.array(clim)
    fin_run = run_avg_per(fin, w=31)

    return fin_run

def get_win_pctl(sst, w=11, p=90, year_length=365):
    """
    Calculate day-of-year w-day window rolling p-percentile and smooth with
    a 31-day periodic moving average.

    Parameters
    ----------
    sst : numpy.ndarray or list
        The input sequence for which the rolling percentile will be calculated.
    w : int, optional, default=11
        The window size for the rolling percentile.
    p : int, optional, default=90
        The percentile value to be calculated within the rolling window.

    Returns
    -------
    fin_run : numpy.ndarray
        The calculated day-of-year rolling percentile smoothed with a 31-day
        periodic moving average.
    """

    if np.isnan(sst[0]):
        return np.array([10000.0] * year_length)

    n = len(sst) // year_length
    var = np.reshape(np.array(sst), (n, year_length))
    hw = w // 2

    # Extend the input array with periodic boundary conditions
    var_ext = np.pad(var, pad_width=((0, 0), (hw, hw)), mode="wrap")

    # Calculate the rolling percentile
    tdh = [np.percentile(var_ext[:, i : i + 2 * hw + 1], p) for i in range(year_length)]

    fin = np.array(tdh)
    fin_run = run_avg_per(fin, w=31)

    return fin_run

def compute_thresholds(
    baseline, window=11, q=90, year_length=365, var="tos", lat="lat", lon="lon"
):
    """
    Compute the rolling q-percentile of the baseline sea surface temperature
    dataset with a specified window size and smooth it using a 31-day periodic moving average.

    Parameters
    ----------
    baseline : xarray.Dataset
        The baseline dataset containing sea surface temperature data.
    window : int, optional, default=11
        The window size for the rolling percentile calculation.
    q : int, optional, default=90
        The percentile value to be calculated within the rolling window.

    Returns
    -------
    ds_thres : xarray.Dataset
        The dataset with the calculated rolling q-percentile smoothed with a 31-day
        periodic moving average.
    """
    baseline_arr = baseline[var].values
    thres = np.apply_along_axis(
        get_win_pctl, 0, baseline_arr, window, q, year_length=year_length
    )
    clim = np.apply_along_axis(
        get_win_clim, 0, baseline_arr, window, year_length=year_length
    )

    baseline = baseline.sortby("time")
    ds_thres = baseline.isel(time=slice(0, year_length))
    ds_thres = ds_thres.groupby("time.dayofyear").mean(dim="time")
    ds_thres = ds_thres.drop_vars(var)
    ds_thres["pctl"] = (("dayofyear", lat, lon), thres)
    ds_thres["clim"] = (("dayofyear", lat, lon), clim)

    ds_thres = ds_thres.where(ds_thres.pctl < 9999)

    return ds_thres

def mhs_to_mhw(mhs, min_days=5, gap=2):
    """
    Convert a marine heat spike (MHS) event sequence to a marine heatwave (MHW) event sequence by applying 
    minimum duration and gap constraints.

    :param mhs: An array-like object of 1s and 0s, with 1s corresponding to sea surface temperature (SST) above the threshold.
    :type mhs: list or numpy.ndarray

    :param min_days: The minimum number of consecutive days required for an MHS event to be considered a MHW event.
    :type min_days: int, optional, default: 5

    :param gap: The maximum allowed gap between two MHW events, in days. If the gap between two MHW events is less than or equal to
                this value, the events will be merged.
    :type gap: int, optional, default: 2

    :return: A numpy array with the same length as the input mhs array, with 1s indicating MHW events and 0s indicating non-MHW events.
    :rtype: numpy.ndarray
    """

    mhs = np.array(mhs)
    split_indices = np.where(np.diff(mhs) != 0)[0] + 1
    split_bool = np.split(mhs, split_indices)
    split = copy.deepcopy(split_bool)

    num_splits = len(split_bool)

    # Handle the first group
    if split_bool[0][0] == 1 and len(split_bool[0]) < min_days:
        split[0] = [0] * len(split_bool[0])

    for i in range(1, num_splits - 1):
        current_group = split_bool[i]
        previous_group = split_bool[i - 1]
        next_group = split_bool[i + 1]

        if (
            current_group[0] == 0
            and len(current_group) <= gap
            and len(previous_group) >= min_days
            and len(next_group) >= min_days
        ):
            split[i] = [1] * len(current_group)
        elif current_group[0] == 1 and len(current_group) < min_days:
            split[i] = [0] * len(current_group)

    # Handle the last group
    if split_bool[-1][0] == 1 and len(split_bool[-1]) < min_days:
        split[-1] = [0] * len(split_bool[-1])

    mhw = np.concatenate(split)

    return mhw

def mhw_duration_1d(arr_1d):
    mhw_durations = []
    mhw_duration = 0
    for day in arr_1d:
        if not np.isnan(day):
            if day == 1:
                mhw_duration += 1
            elif mhw_duration >= 5:
                mhw_durations.append(mhw_duration)
                mhw_duration = 0
            else:
                mhw_duration = 0
        else:
            nan_arr = np.empty(63) 
            nan_arr[:] = np.nan
            return nan_arr

    if mhw_duration >= 5:
        mhw_durations.append(mhw_duration)

    values = np.empty(63)
    values[:] = np.nan
    values[: len(mhw_durations)] = np.array(mhw_durations)

    return np.array(values)

def mhw_mean_anomaly_1d(arr_1d):
    mhw_ans = []
    mhw_an = 0
    i = 0
    for an in arr_1d:
        if not np.isnan(an):
            i += 1
            mhw_an += an
        elif i >= 5:
            mhw_ans.append(mhw_an / i)
            mhw_an = 0
            i = 0
        else:
            mhw_an = 0
            i = 0

    values = np.empty(63)
    values[:] = np.nan
    values[: len(mhw_ans)] = np.array(mhw_ans)

    return np.array(values)

def find_mhw_durations(arr):
    duration_arr = np.apply_along_axis(mhw_duration_1d, 0, arr)
    return duration_arr

def find_mean_anomaly(arr):
    anomaly_arr = np.apply_along_axis(mhw_mean_anomaly_1d, 0, arr)
    return anomaly_arr

def nonzero_and_not_nan(arr):
    # Replace zeros with NaNs temporarily
    arr_with_nans = np.where(arr == 0, np.nan, arr)

    # Find indices of non-NaN elements
    indices = np.argwhere(~np.isnan(arr_with_nans))

    # Extract non-zero, non-NaN elements
    result = [arr_with_nans[idx[0], idx[1], idx[2]] for idx in indices]

    return result