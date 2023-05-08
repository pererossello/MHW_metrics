import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
import cartopy.feature as cfeature
import xarray as xr
import os
import json
import dask
import cmocean
import cartopy.crs as ccrs
import numpy as np


def initialize_figure(fig_size=20, ratio=1, text_size=1, subplots=(1, 1)):
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
        dpi=300,
        layout="constrained",
    )

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig)

    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            ax[i][j] = fig.add_subplot(gs[i, j])
            ax[i][j].grid(which="major", linewidth=fs * 0.015)
            ax[i][j].xaxis.set_tick_params(which="minor", bottom=False)
            ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=1.5 * text_size * fs,
                size=fs * 0.5,
                width=fs * 0.15,
            )

    return fig, ax, fs, text_size


def map_plot(
    ds,
    ax,
    fig,
    extend="max",
    label="",
    fs=1,
    lim=[None, None],
    title="",
    shrink=1,
    cmap="jet",
    cbar=True,
):
    im = ds.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        cmap=cmap,
        vmin=lim[0],
        vmax=lim[1],
    )

    if cbar != False:
        cbar = fig.colorbar(im, orientation="vertical", extend=extend, shrink=shrink)
        cbar.set_label(label, fontsize=fs, labelpad=fs)
        cbar.ax.tick_params(labelsize=fs)
    ax.set_title(title, fontsize=fs)


def err_plot(ax, x_axis, y, y_err, fs, color="k", label_leg="", zorder=0):
    ax.errorbar(
        x_axis,
        y,
        yerr=y_err,
        fmt="s-",
        color=color,
        zorder=zorder,
        elinewidth=fs * 0.1,
        capsize=fs * 0.2,
        capthick=fs * 0.1,
        markersize=fs * 0.2,
        linewidth=fs * 0.1,
        label=label_leg,
    )


def distr_plot(
    ax, distr, fs, color="r", h=1, ref=None, median=False, mean=False, units=""
):
    values = distr["distribution"]
    bins = distr["bins"][:-1]
    if ref == None:
        max = np.max(values)
    else:
        max = np.max(ref)
    values_ = max * np.array(values) / (np.max(values))
    ax.barh(bins, values_, height=h, color=color)

    def find_closest(lst, value):
        index = min(range(len(lst)), key=lambda i: abs(lst[i] - value))
        closest_element = lst[index]
        return index, closest_element

    if median != False:
        idx_median, value_median = find_closest(bins, median)
        ax.hlines(
            bins[idx_median],
            0,
            values[idx_median - 1],
            color="gray",
            linewidth=fs * 0.15,
            zorder=2,
        )

    if mean != False:
        idx_mean, value_mean = find_closest(bins, mean)
        bins_ = bins[idx_mean - 1 : idx_mean + 1]
        ax.hlines(
            bins[idx_mean],
            0,
            values[idx_mean - 1],
            color="k",
            linewidth=fs * 0.15,
            zorder=2,
        )

    custom_lines = [
        Line2D([0], [0], color="k", lw=fs * 0.5),
        Line2D([0], [0], color="gray", lw=fs * 0.5),
    ]

    if mean != False and median != False:
        ax.legend(
            custom_lines,
            [
                f"Mean ({value_mean:0.01f} {units})",
                f"Median ({value_median:0.01f} {units})",
            ],
            loc="upper right",
            fontsize=fs * 0.5,
        )


def plot_horizontal_histogram(
    ax, hist, bin_edges, fs, color="gray", plot_median=False, plot_mean=False, units=""
):
    # Create the bar plot
    ax.barh(
        bin_edges[:-1],
        hist,
        height=np.diff(bin_edges),
        left=0,
        align="edge",
        color=color,
    )

    values = [hist[i] * bin_edges[i] for i in range(len(hist))]
    mean = sum(values) / (sum(hist))

    cumulative_freq = np.cumsum(hist)
    index = np.argmax(cumulative_freq >= 0.5 * np.sum(hist))
    median = bin_edges[index] + (
        0.5 * np.sum(hist) - cumulative_freq[index - 1]
    ) / hist[index] * (bin_edges[index + 1] - bin_edges[index])

    def find_closest(lst, value):
        index = min(range(len(lst)), key=lambda i: abs(lst[i] - value))
        closest_element = lst[index]
        return index, closest_element

    if plot_median != False:
        ax.hlines(
            bin_edges[index],
            0,
            hist[index - 1],
            color="gray",
            linewidth=fs * 0.15,
            zorder=2,
        )
    if plot_mean != False:
        idx, val = find_closest(bin_edges, mean)
        ax.hlines(
            bin_edges[idx], 0, hist[idx], color="black", linewidth=fs * 0.15, zorder=2
        )

    custom_lines = [
        Line2D([0], [0], color="k", lw=fs * 0.4),
        Line2D([0], [0], color="gray", lw=fs * 0.4),
    ]

    if plot_mean != False and plot_median != False:
        ax.legend(
            custom_lines,
            [f"Mean ({mean:0.01f} {units})", f"Median ({median:0.01f} {units})"],
            loc="upper right",
            fontsize=fs * 0.8,
        )


def area(lat, lon):
    """
    Compute area of a rectilinear grid.
    """
    earth_radius = 6371e3
    omega = 7.2921159e-5

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    f = 2 * omega * np.sin(lat_r)
    grad_lon = lon_r.copy()
    grad_lon.data = np.gradient(lon_r)

    dx = grad_lon * earth_radius * np.cos(lat_r)
    dy = np.gradient(lat_r) * earth_radius

    ds_area = xr.DataArray(
        (dx * dy).T, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"]
    )

    return ds_area


def weighted_mean(da):
    ds_area = area(da.lat, da.lon)

    ds_area = ds_area.where(da.isel(time=0).notnull(), drop=True)
    area_sum = np.nansum(np.array(ds_area))
    weighted_area = ds_area / area_sum
    result = (da * weighted_area).sum(("lat", "lon"))

    return result


def mhw_metrics(
    fold,
    ratio=1.65,
    fig_size=25,
    text_size=1.2,
    proportions=[3, 1.2, 2],
    map_lims=[[0, 150], [0, 30], [1.5, 3], [50, 250]],
    savepath="",
):
    ds = xr.open_mfdataset(f"{fold}/*.nc", combine="by_coords").compute()

    distributions = {}
    fold_distr = fold + "distributions/"
    for file in os.listdir(fold_distr):
        with open(f"{fold_distr}/{file}", "r") as infile:
            distributions[file[:-5]] = json.load(infile)
    metrics = list(distributions[file[:-5]].keys())
    distr_keys = list(distributions.keys())
    distributions["total"] = {}
    for metric in metrics:
        distributions["total"][metric] = {
            "hist": np.zeros(len(distributions[distr_keys[0]][metric]["hist"])),
            "bin_edges": distributions[distr_keys[0]][metric]["bin_edges"],
        }
        for key in distr_keys:
            distributions["total"][metric]["hist"] += np.array(
                distributions[key][metric]["hist"]
            )

    distributions = distributions["total"]

    time_series = {
        variable: weighted_mean(ds[variable]) for variable in list(ds.data_vars.keys())
    }

    metrics = ["MHW", "mean_duration", "mean_anomaly", "cumulative_anomaly"]
    hist_metrics = [
        "MHW_days_year",
        "MHW_event_duration",
        "MHW_event_mean_anomaly",
        "MHW_anual_cumulative_anomaly",
    ]

    units = ["days", "days", "$^\circ \!$C", "$^\circ \!$C$\cdot$day"]
    labels = [
        "MHW (days)",
        "MHW dur. (days)",
        "Av. an. ($^\circ \!$C)",
        "Cum. an. ($^\circ \!$C$\cdot$day)",
    ]
    label_legs = ["MHW", "MHW duration", "Average anomaly", "Cumulative anomaly"]

    # map_lims = [[0, 150], [0, 30], [1.5, 3], [50, 250]]
    lims = [[0, 300], [0, 70], [0, 4], [0, 400]]
    x_axis = [i for i in range(int(ds.time.min()), int(ds.time.max()) + 1)]

    cmap = mpl.cm.get_cmap("seismic")
    fillcolors = [cmap(i) for i in [0.6, 0.75, 0.9, 0.95]]
    extent = [
        float(ds.lon.min()),
        float(ds.lon.max()),
        float(ds.lat.min()),
        float(ds.lat.max()),
    ]

    subplots = (4, 3)
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
        dpi=300,
        layout="constrained",
    )
    gs = mpl.gridspec.GridSpec(
        subplots[0], subplots[1], width_ratios=proportions, figure=fig
    )

    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    land = cfeature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="face", facecolor="lightgrey"
    )

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            if j == 2:
                ax[i][j] = fig.add_subplot(gs[i, j], projection=ccrs.Mercator())
                ax[i][j].set_extent(extent, crs=ccrs.PlateCarree())
                ax[i][j].coastlines(linewidth=fs * 0.1, zorder=3)
                ax[i][j].add_feature(land, zorder=2)
                map_plot(
                    ds[metrics[i]].mean("time"),
                    ax=ax[i][2],
                    cmap=cmocean.cm.thermal,
                    fig=fig,
                    fs=fs,
                    label=labels[i],
                    # lim=map_lims[i],
                )
            else:
                ax[i][j] = fig.add_subplot(gs[i, j])
                ax[i][j].grid(which="major", linewidth=fs * 0.015)
                ax[i][j].tick_params(
                    axis="both",
                    which="major",
                    labelsize=text_size * fs,
                    size=fs * 0.5,
                    width=fs * 0.15,
                )
                ax[i][j].xaxis.set_tick_params(which="minor", bottom=True)
                ax[i][j].yaxis.set_minor_locator(AutoMinorLocator())
                ax[i][j].yaxis.set_tick_params(
                    which="minor", left=True, size=fs * 0.3, width=fs * 0.1
                )

            if j == 1:
                ax[i][j].set_xticks([])
                ax[i][j].set_yticklabels([])
                ax[i][j].set_ylim(lims[i])
                colors = [fillcolors[0], "lightgray", "lightgray", "lightgray"]
                plot_horizontal_histogram(
                    ax[i][j],
                    distributions[hist_metrics[i]]["hist"],
                    distributions[hist_metrics[i]]["bin_edges"],
                    fs,
                    color=colors[i],
                    plot_median=True,
                    plot_mean=True,
                    units=units[i],
                )

    pos_err = np.abs(time_series[f"MHS_pos"] - time_series[f"MHS"])
    neg_err = np.abs(time_series[f"MHS_neg"] - time_series[f"MHS"])
    err_plot(
        ax[0][0],
        x_axis,
        time_series["MHS"],
        (pos_err, neg_err),
        fs,
        label_leg="MHS",
        color="gray",
    )

    x_min = ds.time.values[0]
    x_max = ds.time.values[-1]

    ax[0][0].set_title("Yearly mean time-series", fontsize=fs * 1.1)
    ax[0][1].set_title(f"Distribution ({x_min}\u2014{x_max})", fontsize=fs * 1.1)
    ax[0][2].set_title(f"Temporal average ({x_min}\u2014{x_max})", fontsize=fs * 1.1)

    i = 0
    for metric in ["MHW", "mean_duration", "mean_anomaly", "cumulative_anomaly"]:
        pos_err = np.abs(time_series[f"{metric}_pos"] - time_series[metric])
        neg_err = np.abs(time_series[f"{metric}_neg"] - time_series[metric])
        err_plot(
            ax[i][0],
            x_axis,
            time_series[metric],
            (pos_err, neg_err),
            fs,
            label_leg=label_legs[i],
            zorder=1,
        )
        ax[i][0].set_ylim(lims[i])
        ax[i][0].set_xlim([x_min - 0.1, x_max + 0.1])
        ax[i][0].set_xticks(list(range(x_min, x_max, 2)))
        ax[i][0].set_ylabel(labels[i], fontsize=fs)
        ax[i][0].legend(loc="upper left", fontsize=fs)
        if i != 3:
            ax[i][0].set_xticklabels([])

        i += 1

    ax[0][0].fill_between(
        x_axis,
        np.zeros(len(x_axis)),
        time_series["MHW"],
        color=fillcolors[0],
        alpha=1,
        linewidth=0,
        zorder=0,
        label="Moderate",
    )
    ax[0][0].fill_between(
        x_axis,
        np.zeros(len(x_axis)),
        time_series["MHW_cat_2"],
        color=fillcolors[1],
        alpha=1,
        linewidth=0,
        zorder=0,
        label="Strong",
    )
    # ax[0][0].fill_between(
    #     x_axis,
    #     np.zeros(len(x_axis)),
    #     time_series["MHW_cat_3"],
    #     color=fillcolors[3],
    #     alpha=1,
    #     linewidth=0,
    #     zorder=1,
    #     label="Severe",
    # )

    ax[0][0].legend(loc="upper left", fontsize=fs)

    ax[0][0].yaxis.set_minor_locator(AutoMinorLocator())

    plot_horizontal_histogram(
        ax[0][1],
        np.array(distributions["MHW_cat_2_days_year"]["hist"]) / 10,
        distributions["MHW_cat_2_days_year"]["bin_edges"],
        fs,
        color=fillcolors[1],
        plot_median=False,
        plot_mean=False,
    )

    # plot_horizontal_histogram(
    #     ax[0][1],
    #     np.array(distributions["MHW_cat_3_days_year"]["hist"]) / 10,
    #     distributions["MHW_cat_2_days_year"]["bin_edges"],
    #     fs,
    #     color=fillcolors[3],
    #     plot_median=False,
    #     plot_mean=False,
    # )

    if savepath != "":
        plt.savefig(savepath, bbox_inches="tight", dpi=300)


def MHW_year_array(
    ds,
    metric,
    label='',
    fig_size=10,
    text_size=1,
    ratio=2,
    subplots=(2, 5),
    cmap=cmocean.cm.thermal,
):
    land = cfeature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="face", facecolor="lightgrey"
    )

    ds = ds[metric]

    ds_max = np.float32(ds.max().values)
    ds_min = np.float32(ds.min().values)

    x_min = ds.time.values[0]

    extent = [
        float(ds.lon.min()),
        float(ds.lon.max()),
        float(ds.lat.min()),
        float(ds.lat.max()),
    ]

    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(np.sqrt(ratio * fig_size), np.sqrt(fig_size / ratio)),
        dpi=300,
        layout="constrained",
    )

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig)
    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    cmap = cmocean.cm.thermal

    k=0
    for i in range(subplots[0]):
        for j in range(subplots[1]):
            ax[i][j] = fig.add_subplot(gs[i, j], projection=ccrs.Mercator())
            ax[i][j].set_extent(extent, crs=ccrs.PlateCarree())
            ax[i][j].coastlines(linewidth=fs * 0.1, zorder=3)
            ax[i][j].add_feature(land, zorder=2)
            da = ds.isel(time = k)
            map_plot(da, ax[i][j], fig, cmap=cmap, cbar = False)


            mean = np.nanmean(da.values)
            ax[i][j].text(
                0.055,
                0.905,
                "Mean: {:0.02f} {}".format(mean, ''),
                fontsize=fs * text_size,
                transform=ax[i][j].transAxes,
                va="bottom",
                ha="left",
                bbox=dict(facecolor='white', edgecolor='k', alpha = 0.6, lw = 0.1*fs, pad=0.5*fs)
            )

            ax[i][j].set_title(x_min + k, fontsize=2*fs * text_size)

            k+=1

    cbar_ax = fig.add_axes([1.01, 0.05, 0.014, 0.84])
    norm = mpl.colors.Normalize(vmin=ds_min, vmax=ds_max)
    cbar = mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=cmap, extend="both", norm=norm, orientation="vertical"
    )
    cbar.set_label(label, fontsize=fs * text_size * 1.4, rotation=90)
    cbar.ax.tick_params(labelsize=1.4*text_size * fs)