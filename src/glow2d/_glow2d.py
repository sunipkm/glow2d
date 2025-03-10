# %% Imports
from __future__ import annotations
from functools import partial
from typing import List, Optional, SupportsFloat as Numeric, Iterable, Tuple
import warnings
from tqdm import tqdm
import xarray
import xarray as xr
import numpy as np
import glowpython as glow
from datetime import datetime
import pytz
from geopy import Point
from geopy.distance import GreatCircleDistance, EARTH_RADIUS
from haversine import haversine, Unit
import pandas as pd
from scipy.ndimage import geometric_transform
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import simpson
from time import perf_counter_ns
from multiprocessing.pool import Pool

try:
    from ._utils import calc_glow_generic  # relative import
except ImportError:
    from _utils import calc_glow_generic  # for testing

# %%


class glow2d_geo:
    """# 2-D Geocentric GLOW model 
    Evaluate the GLOW model on the great circle passing through the origin location along the specified bearing. 
    The result is presented in a geocentric coordinate system.
    """

    def __init__(self, time: datetime, lat: Numeric, lon: Numeric, heading: Numeric, max_alt: Numeric = 1000, n_pts: int = 50, n_bins: int = 100, *, mpool: Optional[Pool] = None, n_alt: int = None, uniformize_glow: bool = True, tec: Numeric | xarray.Dataset = None, tec_interp_nan: bool = False, full_circ: bool = False, show_progress: bool = True, tqdm_kwargs: dict = None, **kwargs):
        """## Create a GLOWRaycast object.
        This object exposes methods to calculate GLOW model output in GEO coordinates.

        ### Args:
            - `time (datetime)`: Datetime of GLOW calculation.
            - `lat (Numeric)`: Latitude of starting location (degrees).
            - `lon (Numeric)`: Longitude of starting location (degrees).
            - `heading (Numeric)`: Heading (look direction, degrees).
            - `max_alt (Numeric, optional)`: Maximum altitude where intersection is considered. Defaults to 1000.
            - `n_pts (int, optional)`: Number of GEO coordinate angular grid points (i.e. number of GLOW runs), must be even and > 20. Defaults to 50.
            - `n_bins (int, optional)`: Number of GLOW energy bins. Defaults to 100.
            - `mpool (Optional[Pool], optional)`: Multiprocessing pool to be used for multi-threaded evaluation. Defaults to None.
            - `n_alt (int, optional)`: Number of altitude bins, must be > 100. Used only when `uniformize_glow` is set to `True` (default). Defaults to `None`, i.e. uses same number of bins as a single GLOW run.
            - `uniformize_glow (bool, optional)`: Interpolate GLOW output to an uniform altitude grid. `n_alt` is ignored if this option is set to `False`. Defaults to `True`.
            - `tec (Numeric | xarray.Dataset, optional)`: Total Electron Content (TEC) in TECU. If `xarray.Dataset`, it must contain `timestamps`, `gdlat`, and `glon` dimensions. If `None`, TEC is assumed to be 1 TECU. Defaults to `None`.
            - `tec_interp_nan (bool, optional)`: Interpolate NaNs in TEC dataset. Defaults to `False`.
            - `full_circ (bool, optional)`: For testing only, do not use. Defaults to False.
            - `show_progress (bool, optional)`: Use TQDM to show progress of GLOW model calculations. Defaults to True.
            - `tqdm_kwargs (dict, optional)`: Keyword arguments for TQDM. Defaults to None.
            - `kwargs (dict, optional)`: Passed to `glowpython.generic`.

        ### Raises:
            - `ValueError`: Number of position bins can not be odd.
            - `ValueError`: Number of position bins can not be < 20.
            - `ValueError`: Number of altitude bins can not be < 100.
            - `ValueError`: TEC must be a `xarray.Dataset` or a number.
            - `ValueError`: TEC dataset does not contain the provided timestamp, if `tec` is provided as a dataset.
        """
        if n_pts % 2:
            raise ValueError('Number of position bins can not be odd.')
        if n_pts < 20:
            raise ValueError('Number of position bins can not be < 20.')
        self._uniform_glow = uniformize_glow
        self._pt = Point(lat, lon)  # instrument loc
        self._time = time  # time of calc
        if n_alt is not None and n_alt < 100:
            raise ValueError('Number of altitude bins can not be < 100')
        if n_alt is None:
            n_alt = 100  # default
        max_d = 6400 * np.pi if full_circ else EARTH_RADIUS * \
            np.arccos(EARTH_RADIUS / (EARTH_RADIUS + max_alt)
                      )  # find maximum distance where LOS intersects exobase # 6400 * np.pi
        # points on the earth where we need to sample
        self._show_prog = show_progress
        self._kwargs = kwargs
        self._tqdm_kwargs = {} if tqdm_kwargs is None else tqdm_kwargs
        distpts = np.linspace(0, max_d, n_pts, endpoint=True)
        self._locs = []
        self._nbins = n_bins  # number of energy bins (for later)
        for d in distpts:  # for each distance point
            dest = GreatCircleDistance(
                kilometers=d).destination(self._pt, heading)  # calculate lat, lon of location at that distance along the great circle
            self._locs.append((dest.latitude, dest.longitude))  # store it
        if full_circ:  # for fun, _get_angle() does not wrap around for > 180 deg
            npt = self._locs[-1]
            for d in distpts:
                dest = GreatCircleDistance(
                    kilometers=d).destination(npt, heading)
                self._locs.append((dest.latitude, dest.longitude))
        self._angs = self._get_angle()  # get the corresponding angles
        if full_circ:  # fill the rest if full circle
            self._angs[len(self._angs) // 2:] = 2*np.pi - \
                self._angs[len(self._angs) // 2:]
        self._bds = None
        self._mpool = mpool
        if tec is not None:
            if isinstance(tec, xr.Dataset):  # do things if dataset
                tmin = float(tec['timestamps'].min())
                tmax = float(tec['timestamps'].max())
                if not (tmin <= time.timestamp() <= tmax):
                    raise ValueError('TEC dataset does not contain the time %s' % time)
                gdlat = glow.geocent_to_geodet([l[0] for l in self._locs])  # convert to geodetic
                glon = [l[1] for l in self._locs]  # get longitudes
                tecval = []
                for gl, gt in zip(glon, gdlat):
                    tval = tec.interp(coords={'timestamps': time.timestamp(), 'gdlat': gt, 'glon': gl}
                                      ).tec.values.copy()  # interpolate to the time and locations
                    tecval.append(tval)
                tec = np.asarray(tecval)
                if not tec_interp_nan:
                    tec[np.where(np.isnan(tec))] = 1  # replace NaNs with 1
                else:
                    tec = glow.interpolate_nan(tec)  # interpolate NaNs
                self._tec = tec.tolist()  # store the tec
            elif isinstance(tec, Numeric):  # do things if number
                self._tec = np.full(len(self._locs), tec).tolist()  # fill with the number
            else:
                raise ValueError('TEC must be an xarray.Dataset or a number')
        else:
            self._tec = [None]*len(self._locs)  # fill with None

    def _get_angle(self) -> np.ndarray:  # get haversine angles between two lat, lon coords
        angs = []
        for pt in self._locs:
            ang = haversine(self._locs[0], pt, unit=Unit.RADIANS)
            angs.append(ang)
        return np.asarray(angs)

    @staticmethod
    def _uniformize_glow(iono: xarray.Dataset) -> xarray.Dataset:
        alt_km = iono.alt_km.values
        alt = np.linspace(alt_km.min(), alt_km.max(), len(alt_km))  # change to custom
        unit_keys = [
            'O',
            'O2',
            'N2',
            'NO',
            'NS',
            'ND',
            'NeIn',
            'O+',
            'O+(2P)',
            'O+(2D)',
            'O2+',
            'N+',
            'N2+',
            'NO+',
            'N2(A)',
            'N(2P)',
            'N(2D)',
            'O(1S)',
            'O(1D)',
            'NeOut',
            'Te',
            'Ti',
            'Tn',
            'ionrate',
            'pederson',
            'hall',
            'eHeat',
            'Tez'
        ]
        state_keys = ['production', 'loss']
        for key in unit_keys:
            # out = np.interp(alt, alt_km, iono[key].values)
            out = iono[key].values  # in place
            out[np.where(np.isnan(out))] = 0  # replace NaNs with 0
            out = interp1d(alt_km, out, axis=1, fill_value=np.nan)(alt)
            iono[key] = (('angle', 'alt_km'), out, iono[key].attrs)
        ver = iono['ver'].values
        ver[np.where(np.isnan(ver))] = 0
        ver = interp1d(alt_km, ver, axis=1, fill_value=np.nan)(alt)
        iono['ver'] = (('angle', 'alt_km', 'wavelength'), ver, iono['ver'].attrs)
        for key in state_keys:
            out = iono[key].values
            out[np.where(np.isnan(out))] = 0  # replace NaNs with 0
            out = interp1d(alt_km, out, axis=1, fill_value=np.nan)(alt)
            iono[key] = (('angle', 'alt_km', 'state'), out, iono[key].attrs)
        iono['alt_km'] = (('alt_km',), alt, iono['alt_km'].attrs)
        return iono

    # calculate glow model for one location
    def _calg_glow_generic(self, *vars) -> Tuple[int, xarray.Dataset]:
        vars = vars[0]
        lat, lon = vars[0]
        tec = vars[1]
        iono = glow.generic(self._time, lat, lon, self._nbins, tec=tec, **self._kwargs)
        return iono

    def _calc_glow_noprecip(self) -> xarray.Dataset:  # run glow model calculation
        items = zip(self._locs, self._tec)

        if self._mpool is None:
            if self._show_prog:
                pbar = tqdm(items, **self._tqdm_kwargs)
                dss = list(map(self._calg_glow_generic, pbar))
            else:
                dss = list(map(self._calg_glow_generic, items))
        else:
            calcglow = partial(calc_glow_generic, self._time, self._nbins, self._kwargs)
            dss = self._mpool.starmap(calcglow, items)

        # dss.sort(key=lambda x: x[0])
        # dss = list(map(lambda x: x[1], dss))
        self._dss = dss

        tecscale = list(map(lambda x: float(x['tecscale'].values), dss))  # get iriscale

        # for dest in tqdm(self._locs):
        #     self._dss.append(glow.no_precipitation(time, dest[0], dest[1], self._nbins))
        bds: xarray.Dataset = xr.concat(
            self._dss, pd.Index(self._angs, name='angle'))
        bds = bds.drop_dims('tecscale')
        latlon = np.asarray(self._locs)
        bds = bds.assign_coords(lat=('angle', latlon[:, 0]))
        bds = bds.assign_coords(lon=('angle', latlon[:, 1]))
        bds.coords['tecscale'] = xr.Variable(('angle',), tecscale, attrs={
            'description': 'TEC scale factor; non-unity if the IRI TEC is scaled to a provided TEC value.', 'units': 'None', 'long_name': 'TEC scale factor'})

        if self._uniform_glow:
            bds = self._uniformize_glow(bds)
        return bds

    def run_model(self) -> xarray.Dataset:
        """## Run the GLOW model calculation to get the model output in GEO coordinates.

        ### Returns:
            - `xarray.Dataset`: GLOW model output in GEO coordinates.
        """
        if self._bds is not None:
            return self._bds
        # calculate the GLOW model for each lat-lon point determined in init()
        bds = self._calc_glow_noprecip()
        unit_desc_dict = {
            'angle': ('radians', 'Angle of location w.r.t. radius vector at origin (starting location)'),
            'lat': ('degree', 'Latitude of locations'),
            'lon': ('degree', 'Longitude of location')
        }
        _ = list(map(lambda x: bds[x].attrs.update(
            {'units': unit_desc_dict[x][0], 'description': unit_desc_dict[x][1]}), unit_desc_dict.keys()))
        self._bds = bds
        return self._bds  # return the calculated


class glow2d_polar:
    """# 2-D GLOW model in local polar coordinates.
    Use GLOW model output evaluated on a 2D grid using `glow2d_geo` and convert it to a local ZA, R coordinate system at the origin location.
    """

    def __init__(self, bds: xarray.Dataset, altitude: Numeric = 0, *, with_prodloss: bool = False, resamp: Numeric = 1.5):
        """## Create a local polar GLOW2D calculation object.
        Use the GLOW model output in GEO coordinates to calculate the GLOW model output in local polar coordinates.

        ### Args:
            - `bds (xarray.Dataset)`: 2-D GLOW model output in GEO coordinates.
            - `altitude (Numeric, optional)`:  Altitude of local polar coordinate system origin in km above ASL. Must be < 100 km. Defaults to 0.
            - `with_prodloss (bool, optional)`: Calculate production and loss parameters in local coordinates. Defaults to False.
            - `resamp (Numeric, optional)`:  Number of R and ZA points in local coordinate output. ``len(R) = len(alt_km) * resamp`` and ``len(ZA) = n_pts * resamp``. Must be > 0.5. Defaults to 1.5.

        ### Raises:
            - `ValueError`: Resampling can not be < 0.5.
            - `ValueError`: Altitude can not be > 100 km.
        """
        if resamp < 0.5:
            raise ValueError('Resampling can not be < 0.5.')
        if not (0 <= altitude <= 100):
            raise ValueError('Altitude can not be > 100 km.')
        self._resamp = resamp
        self._wprodloss = with_prodloss
        self._bds = bds.copy()
        self._iono = None
        self._r0 = altitude + EARTH_RADIUS

    def transform_coord(self) -> xarray.Dataset:
        """## Execute the coordinate transform 
        Converts GLOW output from GEO to local coordinate system.

        ### Returns:
            - `xarray.Dataset`: GLOW output in (ZA, r) coordinates. This is a reference and should not be modified.
        """
        if self._iono is not None:
            return self._iono
        tt, rr = self.get_local_coords(
            self._bds.angle.values, self._bds.alt_km.values + EARTH_RADIUS, r0=self._r0)  # get local coords from geocentric coords

        self._rmin, self._rmax = self._bds.alt_km.values.min(), rr.max()  # nearest and farthest local pts
        # highest and lowest za
        self._tmin, self._tmax = tt.min(), np.pi / 2  # 0, tt.max()
        self._nr_num = round(len(self._bds.alt_km.values) * self._resamp)  # resample
        self._nt_num = round(len(self._bds.angle.values) * self._resamp)   # resample
        outp_shape = (self._nt_num, self._nr_num)

        # ttidx = np.where(tt < 0)  # angle below horizon (LA < 0)
        # # get distribution of global -> local points in local grid
        # res = np.histogram2d(rr.flatten(), tt.flatten(), range=([rr.min(), rr.max()], [0, tt.max()]))
        # gd = resize(res[0], outp_shape, mode='edge')  # remap to right size
        # gd *= res[0].sum() / gd.sum()  # conserve sum of points
        # window_length = int(25 * self._resamp)  # smoothing window
        # window_length = window_length if window_length % 2 else window_length + 1  # must be odd
        # gd = savgol_filter(gd, window_length=window_length, polyorder=5, mode='nearest')  # smooth the distribution

        self._altkm = altkm = self._bds.alt_km.values  # store the altkm
        self._theta = theta = self._bds.angle.values  # store the angles
        rmin, rmax = self._rmin, self._rmax  # local names
        tmin, tmax = self._tmin, self._tmax
        self._nr = nr = np.linspace(
            rmin, rmax, self._nr_num, endpoint=True)  # local r
        self._nt = nt = np.linspace(
            tmin, tmax, self._nt_num, endpoint=True)  # local look angle
        # get meshgrid of the R, T coord system from regular r, za grid
        self._ntt, self._nrr = self.get_global_coords(nt, nr, r0=self._r0)
        # calculate jacobian
        jacobian = 1  # 4*np.pi*self._nrr*self._nrr / (nr * nr) # self.get_jacobian_glob2loc_glob(self._nrr, self._ntt, r0=self._r0)
        # convert to pixel coordinates
        self._ntt = self._ntt.flatten()  # flatten T, works as _global_from_local LUT
        self._nrr = self._nrr.flatten()  # flatten R, works as _global_from_local LUT
        self._ntt = (self._ntt - self._theta.min()) / \
            (self._theta.max() - self._theta.min()) * \
            len(self._theta)  # calculate equivalent index (pixel coord) from original T grid
        self._nrr = (self._nrr - self._altkm.min() - self._r0) / \
            (self._altkm.max() - self._altkm.min()) * \
            len(self._altkm)  # calculate equivalent index (pixel coord) from original R (alt_km) grid
        # start transformation
        data_vars = {}
        bds = self._bds
        coord_wavelength = bds.wavelength.values  # wl axis
        coord_state = bds.state.values  # state axis
        coord_energy = bds.energy.values  # egrid
        bds_attr = bds.attrs  # attrs
        single_keys = [
            'Tn',
            'Te',
            'Ti',
            'pederson',
            'hall',
        ]  # (angle, alt_km) vars
        density_keys = [
            'O',
            'N2',
            'O2',
            'NO',
            'NS',
            'ND',
            'NeIn',
            'NeOut',
            'ionrate',
            'O+',
            'O+(2P)',
            'O+(2D)',
            'O2+',
            'N+',
            'N2+',
            'NO+',
            'N2(A)',
            'N(2P)',
            'N(2D)',
            'O(1S)',
            'O(1D)',
            'NeOut',
            'eHeat',
            'Tez'
        ]
        state_keys = [
            'production',
            'loss',
        ]  # (angle, alt_km, state) vars
        # start = perf_counter_ns()
        # map all the single key types from (angle, alt_km) -> (la, r)
        for key in single_keys:
            inp = self._bds[key].values
            inp[np.where(np.isnan(inp))] = 0
            out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape, mode='nearest')
            out[np.where(out < 0)] = 0
            # out = warp(inp, inverse_map=(2, self._ntt, self._nrr), output_shape=outp_shape)
            data_vars[key] = (('za', 'r'), out)
        for key in density_keys:
            inp = self._bds[key].values
            inp[np.where(np.isnan(inp))] = 0
            out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape, mode='nearest') / jacobian
            out[np.where(out < 0)] = 0
            # out = warp(inp, inverse_map=(2, self._ntt, self._nrr), output_shape=outp_shape)
            data_vars[key] = (('za', 'r'), out)
        # end = perf_counter_ns()
        # print('Single_key conversion: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        # dataset of (angle, alt_km) vars
        iono = xarray.Dataset(data_vars=data_vars, coords={
            'za': nt, 'r': nr})
        # end = perf_counter_ns()
        # print('Single_key dataset: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        ver = []
        # map all the wavelength data from (angle, alt_km, wavelength) -> (la, r, wavelength)
        for key in coord_wavelength:
            inp = bds['ver'].loc[dict(wavelength=key)].values
            inp[np.where(np.isnan(inp))] = 0
            # scaled by point distribution because flux is conserved, not brightness
            # out = warp(inp, inverse_map=(2, self._ntt, self._nrr), output_shape=outp_shape, mode='nearest') * gd
            out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape, mode='nearest') / jacobian
            out[np.where(out < 0)] = 0
            # inp[ttidx] = 0
            # inpsum = inp.sum()  # sum of input for valid angles
            # outpsum = out.sum()  # sum of output
            # out = out * (inpsum / outpsum)  # scale the sum to conserve total flux
            ver.append(out.T)
        # end = perf_counter_ns()
        # print('VER eval: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        ver = np.asarray(ver).T
        ver = xr.DataArray(
            ver,
            coords={'za': nt, 'r': nr,
                    'wavelength': coord_wavelength},
            dims=['za', 'r', 'wavelength'],
            name='ver'
        )  # create wl dataset
        ver.wavelength.attrs = bds['wavelength'].attrs
        # end = perf_counter_ns()
        # print('VER dataset: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        if self._wprodloss:
            d = {}
            for key in state_keys:  # for each var with (angle, alt_km, state)
                res = []

                def convert_state_stuff(st):
                    inp = bds[key].loc[dict(state=st)].values
                    inp[np.where(np.isnan(inp))] = 0
                    out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape)
                    out[np.where(out < 0)] = 0
                    if key in ('production',):
                        out /= jacobian
                    # out = warp(inp, inverse_map=(2, self._ntt, self._nrr), output_shape=outp_shape)
                    return out.T
                res = list(map(convert_state_stuff, coord_state))
                res = np.asarray(res).T
                d[key] = (('za', 'r', 'state'), res)
            # end = perf_counter_ns()
            # print('Prod_Loss Eval: %.3f us'%((end - start)*1e-3))
            # start = perf_counter_ns()
            prodloss = xarray.Dataset(
                data_vars=d,
                coords={'za': nt, 'r': nr, 'state': coord_state}
            )  # calculate (angle, alt_km, state) -> (la, r, state) dataset
        else:
            prodloss = xarray.Dataset()
        # end = perf_counter_ns()
        # print('Prod_Loss DS: %.3f us'%((end - start)*1e-3))
        ## EGrid conversion (angle, energy) -> (r, energy) ##
        # EGrid is avaliable really at (angle, alt_km = 0, energy)
        # So we get local coords for (angle, R=R0)
        # we discard the angle information because it is meaningless, EGrid is spatial
        # start = perf_counter_ns()
        _rr, _ = self.get_local_coords(
            bds.angle.values, np.ones(bds.angle.values.shape)*self._r0, r0=self._r0)
        _rr = rr[:, 0]  # spatial EGrid
        d = []
        for en in coord_energy:  # for each energy
            inp = bds['precip'].loc[dict(energy=en)].values
            # interpolate to appropriate energy grid
            out = np.interp(nr, _rr, inp)
            d.append(out)
        d = np.asarray(d).T
        precip = xarray.Dataset({'precip': (('r', 'energy'), d)}, coords={
            'r': nr, 'energy': coord_energy})
        precip['energy'].attrs = bds['energy'].attrs
        # end = perf_counter_ns()
        # print('Precip interp and ds: %.3f us'%((end - start)*1e-3))

        # start = perf_counter_ns()
        iono = xr.merge((iono, ver, prodloss, precip))  # merge all datasets
        iono.coords['sflux'] = bds.coords['sflux']
        iono.coords['dwave'] = bds.coords['dwave']
        iono.coords['tecscale'] = bds.coords['tecscale']
        iono.coords['denperturb'] = bds.coords['denperturb']
        iono.coords['altitude'] = (
            ('altitude',), [self._r0 - EARTH_RADIUS],
            {'units': 'km',
             'description': 'Altitude of local polar coordinate origin ASL'}
        )

        _ = list(map(lambda x: iono[x].attrs.update(bds[x].attrs), tuple(iono.data_vars.keys())))  # update attrs from bds

        unit_desc_dict = {
            'za': ('radians', 'Zenith angle'),
            'r': ('km', 'Radial distance in km')
        }
        _ = list(map(lambda x: iono[x].attrs.update(
            {'units': unit_desc_dict[x][0],
             'description': unit_desc_dict[x][1],
             'long_name': unit_desc_dict[x][1]}), unit_desc_dict.keys()))
        
        iono.attrs.update(bds.attrs)
        # end = perf_counter_ns()
        # print('Merging: %.3f us'%((end - start)*1e-3))
        self._iono = iono
        return iono

    @staticmethod
    def get_emission(iono: xarray.Dataset, feature: str = '5577', za_min: Numeric | Iterable = np.deg2rad(20), za_max: Numeric | Iterable = np.deg2rad(25), num_zapts: int = 10, *, rmin: Numeric = None, rmax: Numeric = None, num_rpts: int = 100) -> float | np.ndarray:
        """## Calculate line-of-sight intensity.
        Calculate number of photons per azimuth angle (radians) per unit area per second coming from a region of (`rmin`, `rmax`, `za_min`, `za_max`).

        ### Args:
            - `iono (xarray.Dataset | None)`: GLOW model output in local polar coordinates calculated using `glow2d.glow2d_polar.transform_coord`. If `None`, the function will use the internal `iono` dataset.
            - `feature (str, optional)`: GLOW emission feature. Defaults to '5577'.
            - `za_min (Numeric | Iterable, optional)`: Minimum zenith angle (radians). Defaults to `np.deg2rad(20)`.
            - `za_max (Numeric | Iterable, optional)`: Maximum zenith angle (radians). Defaults to `np.deg2rad(25)`.
            - `num_zapts (int, optional)`: Number of points to interpolate to. Defaults to 10.
            - `rmin (Numeric, optional)`: Minimum distance. Defaults to `None`.
            - `rmax (Numeric, optional)`: Maximum distance. Defaults to `None`.
            - `num_rpts (int, optional)`: Number of distance points. The default is used only if minimum or maximum distance is not `None`. Defaults to 100.

        ### Raises:
            - `ValueError`: `iono` is not an xarray.Dataset.
            - `ValueError`: ZA min and max arrays must be of the same dimension.
            - `ValueError`: ZA min not between 0 deg and 90 deg.
            - `ValueError`: ZA max is not between 0 deg and 90 deg.
            - `ValueError`: ZA min > ZA max.
            - `ValueError`: Selected feature is invalid.

        ### Returns:
            - `float | np.ndarray`: Number of photons/rad/cm^2/s
        """
        if iono is None or not isinstance(iono, xarray.Dataset):
            raise ValueError('iono is not an xarray.Dataset.')
        if isinstance(za_min, Iterable) or isinstance(za_max, Iterable):
            if len(za_min) != len(za_max):
                raise ValueError('ZA min and max arrays must be of the same dimension.')
            callable = partial(glow2d_polar.get_emission, iono=iono, feature=feature,
                               num_zapts=num_zapts, rmin=rmin, rmax=rmax, num_rpts=num_rpts)
            out = list(map(lambda idx: callable(za_min=za_min[idx], za_max=za_max[idx]), range(len(za_min))))
            return np.asarray(out, dtype=float)
        if not (0 <= za_min <= np.deg2rad(90)):
            raise ValueError('ZA must be between 0 deg and 90 deg')
        if not (0 <= za_max <= np.deg2rad(90)):
            raise ValueError('ZA must be between 0 deg and 90 deg')
        if za_min > za_max:
            raise ValueError('ZA min > ZA max')
        if feature not in iono.wavelength.values:
            raise ValueError('Feature %s is invalid. Valid features: %s' % (feature, str(iono.wavelength.values)))
        za: np.ndarray = iono.za.values
        zaxis = iono.za.values
        r: np.ndarray = iono.r.values
        rr = iono.r.values

        if za_min is not None or za_max is not None:
            if (za_min == 0) and (za_max == np.deg2rad(90)):
                pass
            else:
                za_min = za.min() if za_min is None else za_min
                za_max = za.max() if za_max is None else za_max
                zaxis = np.linspace(za_min, za_max, num_zapts, endpoint=True)

        if rmin is not None or rmax is not None:
            rmin = r.min() if rmin is None else rmin
            rmax = r.max() if rmax is None else rmax
            rr = np.linspace(rmin, rmax, num_rpts, endpoint=True)

        ver = iono.ver.loc[dict(wavelength=feature)].values
        ver = (RectBivariateSpline(r, za, ver.T)(rr, zaxis)).T  # interpolate to integration axes

        ver = ver*np.sin(zaxis[:, None])  # integration is VER * sin(phi) * d(phi) * d(r)
        return simpson(simpson(ver.T, zaxis), rr * 1e5)  # do the double integral

    # get global coord index from local coord index, implemented as LUT
    def _global_from_local(self, pt: Tuple[int, int]) -> Tuple[Numeric, Numeric]:
        # if not self.firstrun % 10000:
        #     print('Input:', pt)
        tl, rl = pt  # pixel coord
        # rl = (rl * (self._rmax - self._rmin) / self._nr_num) + self._rmin # real coord
        # tl = (tl * (self._tmax - self._tmin) / self._nt_num) + self._tmin
        # rl = self._nr[rl]
        # tl = self._nt[tl]
        # t, r = self._get_global_coords(tl, rl)
        # # if not self.firstrun % 10000:
        # #     print((rl, tl), '->', (r, t), ':', (self._altkm.min(), self._altkm.max()))
        # r = (r - self._altkm.min() - EARTH_RADIUS) / (self._altkm.max() - self._altkm.min()) * len(self._altkm)
        # t = (t - self._theta.min()) / (self._theta.max() - self._theta.min()) * len(self._theta)
        # if not self.firstrun % 10000:
        #     print((float(r), float(t)))
        return (float(self._ntt[tl*self._nr_num + rl]), float(self._nrr[tl*self._nr_num + rl]))

    @staticmethod
    def get_global_coords(t: np.ndarray | Numeric, r: np.ndarray | Numeric, r0: Numeric = EARTH_RADIUS, meshgrid: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """## Get GEO coordinates from local coordinates.

        $$
            R = \\sqrt{\\left\\{ (r\\cos{\\phi} + R_0)^2 + r^2\\sin{\\phi}^2 \\right\\}}, \\\\
            \\theta = \\arctan{\\frac{r\\sin{\\phi}}{r\\cos{\\phi} + R_0}}
        $$

        ### Args:
            - `t (np.ndarray | Numeric)`: Angles in radians.
            - `r (np.ndarray | Numeric)`: Distance in km.
            - `r0 (Numeric, optional)`: Distance to origin. Defaults to `geopy.distance.EARTH_RADIUS`.
            - `meshgrid (bool, optional)`: Optionally convert 1-D inputs to a meshgrid. Defaults to `True`.

        ### Raises:
            - `ValueError`: ``r`` and ``t`` does not have the same dimensions.
            - `TypeError`: ``r`` and ``t`` are not ``numpy.ndarray``.

        ### Returns:
            - `(np.ndarray, np.ndarray)`: (angles, distances) in GEO coordinates.
        """
        if isinstance(r, np.ndarray) and isinstance(t, np.ndarray):  # if array
            if r.ndim != t.ndim:  # if dims don't match get out
                raise ValueError('r and t does not have the same dimensions')
            if r.ndim == 1 and meshgrid:
                _r, _t = np.meshgrid(r, t)
            elif r.ndim == 1 and not meshgrid:
                _r, _t = r, t
            else:
                _r, _t = r.copy(), t.copy()  # already a meshgrid?
                r = _r[0]
                t = _t[:, 0]
        elif isinstance(r, Numeric) and isinstance(t, Numeric):  # floats
            _r = np.atleast_1d(r)
            _t = np.atleast_1d(t)
        else:
            raise TypeError('r and t must be np.ndarray.')
        # _t = np.pi/2 - _t
        rr = np.sqrt((_r*np.cos(_t) + r0)**2 +
                     (_r*np.sin(_t))**2)  # r, la to R, T
        tt = np.arctan2(_r*np.sin(_t), _r*np.cos(_t) + r0)
        return tt, rr

    @staticmethod
    def get_local_coords(t: np.ndarray | Numeric, r: np.ndarray | Numeric, r0: Numeric = EARTH_RADIUS, meshgrid: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """## Get local coordinates from GEO coordinates.

        $$
            r = \\sqrt{\\left\\{ (R\\cos{\\theta} - R_0)^2 + R^2\\sin{\\theta}^2 \\right\\}}, \\\\
            \\phi = \\arctan{\\frac{R\\sin{\\theta}}{R\\cos{\\theta} - R_0}}
        $$

        ### Args:
            - `t (np.ndarray | Numeric)`: Angles in radians.
            - `r (np.ndarray | Numeric)`: Distance in km.
            - `r0 (Numeric, optional)`: Distance to origin. Defaults to geopy.distance.EARTH_RADIUS.
            - `meshgrid (bool, optional)`: Optionally convert 1-D inputs to a meshgrid. Defaults to True.

        ### Raises:
            - `ValueError`: ``r`` and ``t`` does not have the same dimensions
            - `TypeError`: ``r`` and ``t`` are not ``numpy.ndarray``.

        ### Returns:
            - `(np.ndarray, np.ndarray)`: (angles, distances) in local coordinates.
        """
        if isinstance(r, np.ndarray) and isinstance(t, np.ndarray):
            if r.ndim != t.ndim:  # if dims don't match get out
                raise ValueError('r and t does not have the same dimensions')
            if r.ndim == 1 and meshgrid:
                _r, _t = np.meshgrid(r, t)
            elif r.ndim == 1 and not meshgrid:
                _r, _t = r, t
            else:
                _r, _t = r.copy(), t.copy()  # already a meshgrid?
                r = _r[0]
                t = _t[:, 0]
        elif isinstance(r, Numeric) and isinstance(t, Numeric):
            _r = np.atleast_1d(r)
            _t = np.atleast_1d(t)
        else:
            raise TypeError('r and t must be np.ndarray.')
        rr = np.sqrt((_r*np.cos(_t) - r0)**2 +
                     (_r*np.sin(_t))**2)  # R, T to r, la
        tt = np.arctan2(_r*np.sin(_t), _r*np.cos(_t) - r0)
        return tt, rr

    @staticmethod
    def get_jacobian_glob2loc_loc(r: np.ndarray, t: np.ndarray, r0: Numeric = EARTH_RADIUS) -> np.ndarray:
        """Jacobian \\(|J_{R\\rightarrow r}|\\) for global to local coordinate transform, evaluated at points in local coordinate.

        $$
            |J_{R\\rightarrow r}| = \\frac{R}{r^3}\\left(R^2 + R_0^2 - 2 R R_0 \\cos{\\theta}\\right)
        $$

        ### Args:
            - `r (np.ndarray)`: 2-dimensional array of r.
            - `t (np.ndarray)`: 2-dimensional array of phi.
            - `r0 (Numeric)`: Coordinate transform offset. Defaults to EARTH_RADIUS.

        ### Raises:
            - `ValueError`: Dimension of inputs must be 2.

        ### Returns:
            - `np.ndarray`: Jacobian evaluated at points.
        """
        if r.ndim != 2 or t.ndim != 2:
            raise ValueError('Dimension of inputs must be 2.')
        gt, gr = glow2d_polar.get_global_coords(t, r, r0=r0)
        jac = (gr / (r**3)) * ((gr**2) + (r0**2) - (2*gr*r0*np.cos(gt)))
        return jac

    @staticmethod
    def get_jacobian_loc2glob_loc(r: np.ndarray, t: np.ndarray, r0: Numeric = EARTH_RADIUS) -> np.ndarray:
        """Jacobian \\(|J_{r\\rightarrow R}|\\) for local to global coordinate transform, evaluated at points in local coordinate.

        $$
            |J_{r\\rightarrow R}| = \\frac{r}{R^3}\\left(r^2 + R_0^2 + 2 r R_0 \\cos{\\phi}\\right)
        $$

        ### Args:
            - `r (np.ndarray)`: 2-dimensional array of r.
            - `t (np.ndarray)`: 2-dimensional array of phi.
            - `r0 (Numeric)`: Coordinate transform offset. Defaults to EARTH_RADIUS.

        ### Raises:
            - `ValueError`: Dimension of inputs must be 2.

        ### Returns:
            - `np.ndarray`: Jacobian evaluated at points.
        """
        if r.ndim != 2 or t.ndim != 2:
            raise ValueError('Dimension of inputs must be 2')
        gt, gr = glow2d_polar.get_global_coords(t, r, r0=r0)
        jac = (r/(gr**3))*((r**2) + (r0**2) + (2*r*r0*np.cos(t)))
        return jac

    @staticmethod
    def get_jacobian_glob2loc_glob(gr: np.ndarray, gt: np.ndarray, r0: Numeric = EARTH_RADIUS) -> np.ndarray:
        """Jacobian determinant \\(|J_{R\\rightarrow r}|\\) for global to local coordinate transform, evaluated at points in global coordinate.

        $$
            |J_{R\\rightarrow r}| = \\frac{R}{r^3}\\left(R^2 + R_0^2 - 2 R R_0 \\cos{\\theta}\\right)
        $$

        ### Args:
            - `r (np.ndarray)`: 2-dimensional array of r.
            - `t (np.ndarray)`: 2-dimensional array of phi.
            - `r0 (Numeric)`: Coordinate transform offset. Defaults to EARTH_RADIUS.

        ### Raises:
            - `ValueError`: Dimension of inputs must be 2.

        ### Returns:
            - `np.ndarray`: Jacobian evaluated at points.
        """
        if gr.ndim != 2 or gt.ndim != 2:
            raise ValueError('Dimension of inputs must be 2')
        t, r = glow2d_polar.get_local_coords(gt, gr, r0=r0)
        jac = (gr / (r**3)) * ((gr**2) + (r0**2) - (2*gr*r0*np.cos(gt)))
        return jac

    @staticmethod
    def get_jacobian_loc2glob_glob(gr: np.ndarray, gt: np.ndarray, r0: Numeric = EARTH_RADIUS) -> np.ndarray:
        """Jacobian \\(|J_{r\\rightarrow R}|\\) for global to local coordinate transform, evaluated at points in local coordinate.

        $$
            |J_{r\\rightarrow R}| = \\frac{r}{R^3}\\left(r^2 + R_0^2 + 2 r R_0 \\cos{\\phi}\\right)
        $$

        ### Args:
            - `r (np.ndarray)`: 2-dimensional array of r.
            - `t (np.ndarray)`: 2-dimensional array of phi.
            - `r0 (Numeric)`: Coordinate transform offset. Defaults to EARTH_RADIUS.

        ### Raises:
            - `ValueError`: Dimension of inputs must be 2.

        ### Returns:
            - `np.ndarray`: Jacobian evaluated at points.
        """
        if gr.ndim != 2 or gt.ndim != 2:
            raise ValueError('Dimension of inputs must be 2')
        t, r = glow2d_polar.get_local_coords(gt, gr, r0=EARTH_RADIUS)
        jac = (r/(gr**3))*((r**2) + (r0**2) + (2*r*r0*np.cos(t)))
        return jac


def geo_model(time: datetime, lat: Numeric, lon: Numeric, heading: Numeric, max_alt: Numeric = 1000, n_pts: int = 50, n_bins: int = 100, *, mpool: Optional[Pool] = None, n_alt: int = None, tec: Numeric | xarray.Dataset = None, tec_interp_nan: bool = False, show_progress: bool = True, tqdm_kwargs: dict = None, **kwargs) -> xarray.Dataset:
    """## Evaluate 2-D GLOW model in GEO coordinates
    Run GLOW model looking along heading from the current location and return the model output in
    (T, R) geocentric coordinates where T is angle in radians from the current location along the great circle
    following current heading, and R is altitude in kilometers. R is in an uniform grid with `n_alt` points.

    ### Args:
        - `time (datetime)`: Datetime of GLOW calculation.
        - `lat (Numeric)`: Latitude of starting location (degrees).
        - `lon (Numeric)`: Longitude of starting location (degrees).
        - `heading (Numeric)`: Heading (look direction, degrees).
        - `max_alt (Numeric, optional)`: Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
        - `n_pts (int, optional)`: Number of GEO coordinate angular grid points (i.e. number of GLOW runs), must be even and > 20. Defaults to 50.
        - `n_bins (int, optional)`: Number of energy bins. Defaults to 100.
        - `mpool (Optional[Pool], optional)`: Multiprocessing pool to be used for multi-threaded evaluation. Defaults to `None`.
        - `n_alt (int, optional)`: Number of altitude bins, must be > 100. Defaults to None, i.e. uses same number of bins as a single GLOW run.
        - `tec (Numeric | xarray.Dataset, optional)`: Total Electron Content (TEC) in TECU. If `xarray.Dataset`, it must contain `timestamps`, `gdlat`, and `glon` dimensions. If `None`, TEC is assumed to be 1 TECU. Defaults to `None`.
        - `tec_interp_nan (bool, optional)`: Interpolate NaN values in TEC dataset. Defaults to `False`.
        - `show_progress (bool, optional)`: Use TQDM to show progress of GLOW model calculations. Defaults to True.
        - `tqdm_kwargs (dict, optional)`: Passed to `tqdm.tqdm`. Defaults to None.
        - `kwargs (dict, optional)`: Passed to `glowpython.generic`.

    ### Raises:
        - `ValueError`: Number of position bins can not be odd.
        - `ValueError`: Number of position bins can not be < 20.
        - `ValueError`: Resampling can not be < 0.5.

    ### Returns:
        - `xarray.Dataset`: Ionospheric parameters and brightnesses in GEO coordinates.
    """
    grobj = glow2d_geo(time, lat, lon, heading, max_alt,
                       n_pts, n_bins, mpool=mpool,
                       n_alt=n_alt, uniformize_glow=True,
                       tec=tec, tec_interp_nan=tec_interp_nan,
                       show_progress=show_progress,
                       tqdm_kwargs=tqdm_kwargs, **kwargs)
    bds = grobj.run_model()
    return bds


def polar_model(time: datetime, lat: Numeric, lon: Numeric, heading: Numeric, altitude: Numeric = 0, max_alt: Numeric = 1000, n_pts: int = 50, n_bins: int = 100, *, mpool: Optional[Pool] = None, n_alt: int = None, tec: Numeric | xarray.Dataset = None, tec_interp_nan: bool = False, with_prodloss: bool = False, full_output: bool = False, resamp: Numeric = 1.5, show_progress: bool = True, tqdm_kwargs: dict = None, **kwargs) -> xarray.Dataset | Tuple[xarray.Dataset, xarray.Dataset]:
    """## Evaluate 2-D GLOW model in local polar coordinates
    Run GLOW model looking along heading from the current location and return the model output in
    (ZA, R) local coordinates where ZA is zenith angle in radians and R is distance in kilometers.

    ### Args:
        - `time (datetime)`: Datetime of GLOW calculation.
        - `lat (Numeric)`: Latitude of starting location (degrees).
        - `lon (Numeric)`: Longitude of starting location (degrees).
        - `heading (Numeric)`: Heading (look direction, degrees).
        - `altitude (Numeric, optional)`: Altitude of local polar coordinate system origin in km above ASL. Must be < 100 km. Defaults to 0.
        - `max_alt (Numeric, optional)`: Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
        - `n_pts (int, optional)`: Number of GEO coordinate angular grid points (i.e. number of GLOW runs), must be even and > 20. Defaults to 50.
        - `n_bins (int, optional)`: Number of energy bins. Defaults to 100.
        - `mpool (Optional[Pool], optional)`: Multiprocessing pool to be used for multi-threaded evaluation. Defaults to `None`.
        - `n_alt (int, optional)`: Number of altitude bins, must be > 100. Defaults to `None`, i.e. uses same number of bins as a single GLOW run.
        - `tec (Numeric | xarray.Dataset, optional)`: Total Electron Content (TEC) in TECU. If `xarray.Dataset`, it must contain `timestamps`, `gdlat`, and `glon` dimensions. If `None`, TEC is assumed to be 1 TECU. Defaults to `None`.
        - `tec_interp_nan (bool, optional)`: Interpolate NaN values in TEC dataset. Defaults to `False`.
        - `with_prodloss (bool, optional)`: Calculate production and loss parameters in local coordinates. Defaults to `False`.
        - `full_output (bool, optional)`: Returns only local coordinate GLOW output if `False`, and a tuple of local and GEO outputs if `True`. Defaults to `False`.
        - `resamp (Numeric, optional)`: Number of R and ZA points in local coordinate output. ``len(R) = len(alt_km) * resamp`` and ``len(ZA) = n_pts * resamp``. Must be > 0.5. Defaults to 1.5.
        - `show_progress (bool, optional)`: Use TQDM to show progress of GLOW model calculations. Defaults to `True`. Does not apply if `mpool` is not `None`.
        - `tqdm_kwargs (dict, optional)`: Passed to `tqdm.tqdm`. Defaults to `None`.
        - `kwargs (dict, optional)`: Passed to `glowpython.generic`.

    ### Returns:
        - `xarray.Dataset | Tuple[xarray.Dataset, xarray.Dataset]`: Ionospheric parameters and brightnesses (with or without production and loss) in local coordinates. This is a reference and should not be modified.

        - `iono, bds (xarray.Dataset, xarray.Dataset)`: These values are returned only if `full_output == True`. Both are references and should not be modified.
            - Ionospheric parameters and brightnesses (with or without production and loss) in local coordinates.
            - Ionospheric parameters and brightnesses (with production and loss) in GEO coordinates.

    ### Raises:
        - `ValueError`: Number of position bins can not be odd.
        - `ValueError`: Number of position bins can not be < 20.
        - `ValueError`: n_alt can not be < 100.
        - `ValueError`: Resampling can not be < 0.5.
        - `ValueError`: altitude must be in the range [0, 100].

    """
    grobj = glow2d_geo(time, lat, lon, heading, max_alt, n_pts, n_bins,
                       mpool=mpool, tec=tec, tec_interp_nan=tec_interp_nan,
                       n_alt=n_alt, uniformize_glow=True,
                       show_progress=show_progress,
                       tqdm_kwargs=tqdm_kwargs, **kwargs)
    bds = grobj.run_model()
    grobj = glow2d_polar(bds, altitude, with_prodloss=with_prodloss, resamp=resamp)
    iono = grobj.transform_coord()
    if not full_output:
        return iono
    else:
        return (iono, bds)


# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from multiprocessing import Pool as init_pool, cpu_count

    time = datetime(2022, 3, 22, 18, 0).astimezone(pytz.utc)
    print('Evaluation time:', time)
    lat, lon = 42.64981361744372, -71.31681056737486
    bdss: List[xr.Dataset] = []
    for mt in [None, init_pool(cpu_count())]:
        grobj = glow2d_geo(time, 42.64981361744372, -71.31681056737486, 40, n_pts=100,
                           mpool=mt,
                           show_progress=False,
                           tqdm_kwargs={'desc': 'GLOW GEO'})
        st = perf_counter_ns()
        bds = grobj.run_model()
        end = perf_counter_ns()
        print(f'Time to generate: {(end - st)*1e-6:.6f} ms')
        st = perf_counter_ns()
        grobj = glow2d_polar(bds, with_prodloss=False, resamp=1)
        iono = grobj.transform_coord()
        end = perf_counter_ns()
        print(f'Time to convert: {(end - st)*1e-6:.6f} ms')
        print()
        feature = '5577'
        print(f'Number of photons between 70 - 75 deg ZA ({feature} A):',
              grobj.get_emission(iono, feature=feature, za_min=np.deg2rad(70), za_max=np.deg2rad(75)))

        za_min = np.arange(0, 90, 2.5, dtype=float)
        za_max = za_min + 2.5
        za = za_min + 1.25
        za_min = np.deg2rad(za_min)
        za_max = np.deg2rad(za_max)
        pc = grobj.get_emission(iono, za_min=za_min, za_max=za_max)
        plt.title('Altitude Angle vs. Photon Count Rate (5577 A)')
        plt.plot(pc, 90 - za)
        plt.xscale('log')
        plt.ylabel('Altitude Angle (deg)')
        plt.xlabel(r'Photon Count Rate (cm$^{-2}$ rad$^{-1}$ s$^{-1}$)')
        plt.ylim(0, 90)
        # plt.xlim(pc.min(), pc.max())
        plt.xlim(1e6, pc.max())
        plt.show()
        bds.ver.loc[{'wavelength': '5577'}].plot()
        plt.show()
        iono['N2+'].plot()
        plt.show()
        bdss.append(bds)
    assert bdss[0].equals(bdss[1])

# %%
