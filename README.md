# GLOW2D

This package builds off of the GLobal airglOW model. The GLOW model is evaluated on an altitude grid at a location. For non-zenith observation geometries, the model requires evaluation at multiple points along the line of sight. The `glow2d` library wraps the `ncarglow.no_precipitation()` function to generate the output on a 2D grid in a geocentric (GEO) coordinate system and in a local (ZA, R) coordinate system.

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
- [ncarglow](https://github.com/sunipkm/ncar-glow)
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

The `iono_loc` dataset is:

```
<xarray.Dataset>
Dimensions:     (za: 200, r: 500, wavelength: 15, energy: 100)
Coordinates:
  * za          (za) float64 1.571 1.563 1.555 1.547 ... 0.01579 0.007893 0.0
  * r           (r) float64 0.0 7.391 14.78 ... 3.673e+03 3.681e+03 3.688e+03
  * wavelength  (wavelength) <U21 '3371' '4278' '5200' ... '1356' '1493' '1304'
  * energy      (energy) float64 0.2 0.8 1.2 1.8 2.2 ... 457.3 480.7 505.4 531.3
Data variables: (12/18)
    Tn          (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    O           (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    N2          (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    O2          (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    NO          (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    NeIn        (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    ...          ...
    pedersen    (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    hall        (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    Te          (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    Ti          (za, r) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
    ver         (za, r, wavelength) float64 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
    precip      (r, energy) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
Attributes:
    geomag_params:                       Ap     Kp   f107 resolution       f1...
    time:           2022-02-15T11:00:00+00:00
    glatlon:        (42.64981361744372, -71.31681056737486)
```

The `iono_geo` dataset is:
```
<xarray.Dataset>
Dimensions:         (angle: 100, alt_km: 250, wavelength: 15, state: 12,
                     energy: 100)
Coordinates:
  * alt_km          (alt_km) float64 60.0 60.6 61.2 61.8 ... 953.4 957.9 962.4
  * wavelength      (wavelength) <U21 '3371' '4278' '5200' ... '1493' '1304'
  * state           (state) <U6 'O+(2P)' 'O+(2D)' 'O+(4S)' ... 'O(1S)' 'O(1D)'
  * energy          (energy) float64 0.2 0.8 1.2 1.8 ... 457.3 480.7 505.4 531.3
  * angle           (angle) float64 0.0 0.005323 0.01065 ... 0.5163 0.5217 0.527
    lat             (angle) float64 42.65 42.88 43.12 ... 60.15 60.25 60.34
    lon             (angle) float64 -71.32 -71.05 -70.78 ... -31.7 -31.12 -30.53
Data variables: (12/21)
    Tn              (angle, alt_km) float64 237.0 236.0 235.0 ... 815.0 815.0
    O               (angle, alt_km) float64 0.0 0.0 0.0 ... 1.2e+03 1.11e+03
    N2              (angle, alt_km) float64 4.49e+15 4.17e+15 ... 0.0102 0.00893
    O2              (angle, alt_km) float64 1.2e+15 1.12e+15 ... 1.9e-05
    NO              (angle, alt_km) float64 7.81e+07 7.81e+07 ... 5.92e-08
    NeIn            (angle, alt_km) float64 100.0 100.0 ... 2.35e+04 2.34e+04
    ...              ...
    Ti              (angle, alt_km) float64 236.9 235.9 ... 2.715e+03 2.726e+03
    ver             (angle, alt_km, wavelength) float64 0.0 0.0 0.0 ... 0.0 0.0
    production      (angle, alt_km, state) float64 0.0 nan nan ... 0.0 0.0 0.0
    loss            (angle, alt_km, state) float64 nan nan nan ... 0.0 1.1 0.01
    precip          (angle, energy) float64 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
    excitedDensity  (angle, alt_km, state) float64 nan nan nan ... 0.0 0.0 0.01
...
Attributes:
    geomag_params:                       Ap     Kp   f107 resolution       f1...
    time:           2022-02-15T11:00:00+00:00
    glatlon:        (42.64981361744372, -71.31681056737486)
```

All angles are in radians, and distances in km. Due to the nature of the coordinate transformation, local coordinate outputs do not contain any `NaN` values.