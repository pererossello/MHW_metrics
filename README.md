# MHW Metrics package

'MHW_metrics' is a Python package that enables easy computation of yearly averages of Marine Heat Wave (MHW) metrics from Sea Surface Temperature (SST) netCDF4 files. The package is based on the MHW definition from Hobday et al. (2016) and was initially created for the satellite data analysis of Rosselló et al. (2023). The package is primarily intended for Copernicus Marine Service (CMEMS) datasets such as SST_GLO_SST_L4_REP_OBSERVATIONS_010_011 and SST_MED_SST_L4_REP_OBSERVATIONS_010_021, but it may also be compatible with other daily SST datasets.

## Features

- For a given netCDF4 file containing daily SST data for a specific region, compute the following per grid cell and per year:

    - MHW days
    - Marine Heat Spikes (MHS)
    - Mean MHW SST anomaly
    - Cumulative MHW SST anomaly
    - Mean MHW duration
- Compute the yearly distribution of these metrics for each MHW event in the region. For example, for each year, calculate the duration of all MHW events that have occurred in the region.

- If there is a daily interpolation error in the SST time series, it can be incorporated to calculate upper and lower bounds of MHW metrics considering such error.


## Installation

```bash
pip install git+https://github.com/canagrisa/MHW_metrics.git
```

## Usage

```python 

#TBD


```

## License

MIT License. See the LICENSE file for more information.

## Support and Contributions

For support or to report bugs, please open a GitHub issue. Contributions are welcome! Feel free to submit a pull request or suggest new features by opening an issue.

## Acknowledgments

This package uses the MHW definition and methodology of Hobday et al. (2016) and Hobday et al. (2018).
