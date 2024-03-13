# GLOW2D

This package builds off of the GLobal airglOW model. The GLOW model is evaluated on an altitude grid at a location. For non-zenith observation geometries, the model requires evaluation at multiple points along the line of sight. The `glow2d` library wraps the `glowpython.generic()` function to generate the output on a 2D grid in a geocentric (GEO) coordinate system and in a local (ZA, R) coordinate system.

## Installation
Directly using `pip`:
```sh
$ pip install glow2d
```

Indirectly by `git clone`:
```sh
$ git clone https://github.com/sunipkm/glow2d && cd glow2d
$ pip install .
```

## Dependencies
The following non-trivial dependencies are present:
- [glowpython](https://github.com/sunipkm/glowpython)
- [geomagdata](https://pypi.org/projects/geomagdata)
- [geopy](https://pypi.org/project/geopy/)
- [haversine](https://pypi.org/project/haversine/)
- [xarray](https://pypi.org/project/xarray/)

This code uses [glowpython](https://pypi.org/project/glowpython/) package to run the GLOW model.
[geomagdata](https://pypi.org/project/geomagdata/) is used to get the $F_{10.7}$ and $A_p$ indices from the [new data source](https://www.gfz-potsdam.de/en/kp-index/).

## Usage
For example:

```py
from datetime import datetime
from glow2d import glow2d_polar as grc

time = datetime(2022, 2, 15, 20, 0, 0).astimezone('UTC') # 20:00 hours local in UTC
lat, lon = 42.64981361744372, -71.31681056737486 # location of observation
bearing = 40 # 40 deg northeast, refer to geopy.GreatCircleDistance().destination() for more info.

iono_loc, iono_geo = grc.polar_model(time, lat, lon, bearing, full_output=True) # generate output in local and GEO coordinates
```

The output datasets are [xarray](https://pypi.org/project/xarray/) Datasets with the following structure:
- Coordinates:
    - `alt_km`: Altitude grid (`km`) [only in GEO coordinates, from ASL]
    - `za`: Zenith angle (`radians`) [only in local polar coordinates]
    - `r`: Radial distance from observation point (`km`) [only in local polar coordinates]
    - `angle`: Angle along great circle (`radians`)
    - `energy`: Energy grid (`eV`)
    - `state`: Neutral/Ionic excitation states (`string`) [only if production and loss are converted in local polar coordinates]
    - `wavelength`: Wavelength of emission features in Angstrom (`string`). Represented as strings to accommodate the LBH band.
    - `lat`: Latitude (`degrees`)
    - `lon`: Longitude (`degrees`)
    - `wave`: Solar flux wavelength (`Angstrom`)
    - `dwave`: Solar flux wavelength bin width (`Angstrom`)
    - `sflux`: Solar flux (`W/m^2/Angstrom`)
    - `tecscale`: TEC scale factor (`float`)
    - `altitude`: Altitude of observation point (`km` from ASL) [only in local polar coordinates]
- Data:
    - Dimension (`(angle, alt_km)` in GEO, `(angle, za)` in polar):
        - `O`: Neutral atomic oxygen density (`cm^-3`)
        - `O2`: Neutral molecular oxygen density (`cm^-3`)
        - `N2`: Neutral molecular nitrogen density (`cm^-3`)
        - `NO`: Neutral nitric oxide density (`cm^-3`)
        - `NS`: N(4S) density (`cm^-3`)
        - `ND`: N(2D) density (not used, set to zero) (`cm^-3`)
        - `NeIn`: Electron density (IRI-90), input to GLOW radiative transfer model. (`cm^-3`)
        - `O+`: Ion atomic oxygen (4S) density (`cm^-3`)
        - `O+(2P)`: Ion atomic oxygen (2P) density (`cm^-3`)
        - `O+(2D)`: Ion atomic oxygen (2D) density (`cm^-3`)
        - `O2+`: Ion molecular oxygen density (`cm^-3`)
        - `N+`: Ion molecular nitrogen density (`cm^-3`)
        - `N2+`: Ion molecular nitrogen density (`cm^-3`)
        - `NO+`: Ion nitric oxide density (`cm^-3`)
        - `N2(A)`: Molecular nitrogen (GLOW) density (`cm^-3`)
        - `N(2P)`: N(2P) density (`cm^-3`)
        - `N(2D)`: N(2D) density (`cm^-3`)
        - `O(1S)`: O(1S) density (`cm^-3`)
        - `O(1D)`: O(1D) density (`cm^-3`)
        - `NeOut`: Electron density (calculated below 200 km for `kchem=4` using GLOW model, `cm^-3`) 
        - `Te`: Electron temperature (`K`)
        - `Ti`: Ion temperature (`K`)
        - `Tn`: Neutral temperature (`K`)
        - `ionrate`: Ionization rate (`1/s`)
        - `pederson`: Pederson conductivity (`S/m`)
        - `hall`: Hall conductivity (`S/m`)
        - `eHeat`: Ambient electron heating rate (`eV/cm^3/s`)
        - `Tez`: Total energetic electron energy deposition (`eV/cm^3/s`)
    - Dimension ((`angle`, `alt_km`, `wavelength`) in GEO, (`angle`, `za`, `wavelength`) in polar):
        - `ver`: Volume emission rate of various features (`1/cm^3/s`)
    - Dimension (`angle`, `alt_km`, `state`) in GEO, (`angle`, `za`, `state`) in polar:
        - `production`: Production rate of various species  (`1/cm^3/s`)
        - `loss`: Fractional loss rate of various species (`1/s`)
    - Dimension (`energy`):
        - `precip`: Precipitation flux (`cm^-2/s/eV`)
- Attributes:
    - `time`: Time of evaluation (`ISO 8601 formatted string`)
    - `glatlon`: Geographic latitude and longitude of evaluation (`degrees`)
    - `Q`: Flux of precipitating electrons (`erg/cm^2/s`)
    - `Echar`: Characteristic energy of precipitating electrons (`eV`)
    - `f107a`: 81-day average of F10.7 solar flux (`FSU`)
    - `f107`: Present day F10.7 solar flux (`FSU`)
    - `f107p`: Previous day F10.7 solar flux (`FSU`)
    - `ap`: Magnetic index
    - `iscale`: Solar flux model. `0`: Hinteregger, `1`: EUVAC
    - `xuvfac`: XUV enhancement factor. 
    - `itail`: Low energy tail enabled/disabled.
    - `fmono`: Monoenergetic energy flux (`erg/cm^2`).
    - `emono`: Monoenergetic characteristic energy (`keV`).
    - `jlocal`: Local calculation only (disable electron transport).
    - `kchem`: GLOW chemistry level.

All angles are in radians, and distances in km. Due to the nature of the coordinate transformation, local coordinate outputs do not contain any `NaN` values.