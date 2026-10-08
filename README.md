# Slip–coupling sampling

Code to compare finite-fault slip models with kinematic coupling models, accompanying:

> Oryan, B. & Gabriel, A.-A. *Do Coupled Megathrusts Rupture?* Preprint, [doi:10.31223/X5HB3N](https://doi.org/10.31223/X5HB3N).

This repository shows how to sample finite-fault slip models and compute slip–coupling correlations as in Oryan & Gabriel. Each finite-fault model is turned into a point cloud. Every subfault $i$ gets $N_i = \kappa A_i S_i$ points, drawn uniformly within the subfault, so the density follows slip. Each point then takes the coupling value at its location, and the empirical CDF of those values shows how slip is distributed across coupling. A null model, built by relocating and rotating the rupture at random within the margin, gives the slip–coupling relation expected by chance.

You are welcome to apply this code to your own earthquakes and coupling models. All coupling models can be downloaded from the [Coupling Cloud](https://couplingcloud.ucsd.edu), and finite-fault models from [SRCMOD](http://equake-rc.info/srcmod/).

## Repository

| File | Contents |
|---|---|
| `eventLoader.py` | Reader for SRCMOD / USGS `.fsp` finite-fault files |
| `finiteFaultSampler.py` | Fault segments and subfaults, slip-weighted sampling, shifting and rotating ruptures |
| `couplingPlotter.py` | Coupling grids (`CouplingObjectGrid`), coupling sampler, map plotting |
| `torchOperators.py` | Grid interpolation and polygon samplers (PyTorch) |
| `cdf_pdf.py` | Chunked (dask) ECDFs and KDEs over many realizations |
| `examples/chileSlipCouplingExample.ipynb` | Worked example: 2007 Mw 7.7 Tocopilla, Chile (Fig. 2) |

## Requirements

Python ≥ 3.10 and:

```
numpy pandas scipy matplotlib seaborn python-dateutil
torch mapbox_earcut
pyproj shapely geopandas cartopy
xarray netCDF4
dask distributed scikit-learn
jupyter
```

```bash
pip install numpy pandas scipy matplotlib seaborn python-dateutil torch mapbox_earcut \
            pyproj shapely geopandas cartopy xarray netCDF4 dask distributed scikit-learn jupyter
```

The first time cartopy draws coastlines it downloads Natural Earth data, so that run needs an internet connection.

## Quick start

```bash
cd examples
jupyter notebook chileSlipCouplingExample.ipynb
```

The notebook loads the coupling and slip models, samples the rupture with one κ, and plots the slip-weighted coupling CDF. It shows how to choose a converged κ (Supplementary Text S2), then builds a small null model from 5 randomly relocated ruptures and computes all the CDFs with `cdf_pdf`.

## Data

- **Coupling models:** The Coupling Cloud, [couplingcloud.ucsd.edu](https://couplingcloud.ucsd.edu) (Oryan et al., 2026, *Seismica*, [doi:10.26443/seismica.v5i1.2314](https://doi.org/10.26443/seismica.v5i1.2314)). The example uses Métois et al. (2016) (`examples/Metois.nc`).
- **Finite-fault models:** SRCMOD (Mai & Thingbaijam, 2014, [equake-rc.info/srcmod](http://equake-rc.info/srcmod/)). The example uses `s2007TOCOPI01HAYE` (Hayes, 2017).

All data for the paper are available in *Supplements for Do Coupled Megathrusts Rupture?*, [doi:10.5281/zenodo.22836382](https://doi.org/10.5281/zenodo.22836382).

## Citation

If you use this code, please cite the preprint above.
