# %% Imports
from __future__ import annotations
import typing
import xarray as xr
import numpy as np
import ncarglow as glow
from datetime import datetime
import pytz
from geopy import Point
from geopy.distance import GreatCircleDistance, EARTH_RADIUS
from haversine import haversine, Unit
import pandas as pd
from tqdm.contrib.concurrent import thread_map, process_map
from scipy.ndimage import geometric_transform
from time import perf_counter_ns
import platform

MAP_FCN = process_map
if platform.system() == 'Darwin':
    MAP_FCN = thread_map

__version__ = '0.1.0'

# %%
class GLOWRaycast:
    __version__ = __version__
    def __init__(self, time: datetime, lat: float, lon: float, heading: float, max_alt: float = 1000, n_pts: int = 50, n_bins: int = 100, *, with_prodloss: bool = False, n_threads: int = 24, full_circ: bool = False):
        """Create a GLOWRaycast object.

        Args:
            time (datetime): Datetime of GLOW calculation.
            lat (float): Latitude of starting location.
            lon (float): Longitude of starting location.
            heading (float): Heading (look direction).
            max_alt (float, optional): Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
            n_pts (int, optional): Number of GEO coordinate angular grid points (i.e. number of GLOW runs). Defaults to 200.
            n_bins (int, optional): Number of energy bins. Defaults to 100.
            with_prodloss (bool, optional): Calculate production and loss parameters in local coordinates. Default: False
            n_threads (int, optional): Number of threads for parallel GLOW runs. Defaults to 24.
            full_circ (bool, optional): For testing only, do not use. Defaults to False.
        """
        self._wprodloss = with_prodloss
        self._pt = Point(lat, lon)  # instrument loc
        self._time = time  # time of calc
        max_d = 6400 * np.pi if full_circ else EARTH_RADIUS * \
            np.arccos(EARTH_RADIUS / (EARTH_RADIUS + max_alt)
                      )  # find maximum distance where LOS intersects exobase # 6400 * np.pi
        # points on the earth where we need to sample
        distpts = np.linspace(0, max_d, n_pts, endpoint=True)
        self._locs = []
        self._nbins = n_bins  # number of energy bins (for later)
        self._nthr = n_threads  # number of threads (for later)
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
        self._iono = None

    @classmethod
    def no_precipitation(cls, time: datetime, lat: float, lon: float, heading: float, max_alt: float = 1000, n_pts: int = 50, n_bins: int = 100, *, with_prodloss=False, n_threads: int = 24, full_output: bool = False) -> typing.Any:
        """Run GLOW model looking along heading from the current location and return the model output in
        (ZA, R) local coordinates where ZA is zenith angle in radians and R is distance in kilometers.

        Args:
            time (datetime): Datetime of GLOW calculation.
            lat (float): Latitude of starting location.
            lon (float): Longitude of starting location.
            heading (float): Heading (look direction).
            max_alt (float, optional): Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
            n_pts (int, optional): Number of GEO coordinate angular grid points (i.e. number of GLOW runs). Defaults to 200.
            n_bins (int, optional): Number of energy bins. Defaults to 100.
            with_prodloss (bool, optional): Calculate production and loss parameters in local coordinates. Default: False
            n_threads (int, optional): Number of threads for parallel GLOW runs. Defaults to 24.
            full_output (bool, optional): Returns only local coordinate GLOW output if False, and a tuple of local and GEO outputs if True. Defaults to False.
        """
        grobj = cls(time, lat, lon, heading, max_alt, n_pts, n_bins,
                    n_threads=n_threads, with_prodloss=with_prodloss)
        bds = grobj.run_no_precipitation()
        iono = grobj.transform_coord()
        if not full_output:
            return iono
        else:
            return (iono, bds)

    @classmethod
    def no_precipitation_geo(cls, time: datetime, lat: float, lon: float, heading: float, max_alt: float = 1000, n_pts: int = 50, n_bins: int = 100, *, n_threads: int = 24) -> xr.Dataset:
        """Run GLOW model looking along heading from the current location and return the model output in
        (T, R) geocentric coordinates where T is angle in radians from the current location along the great circle
        following current heading, and R is altitude in kilometers.

        Args:
            time (datetime): Datetime of GLOW calculation.
            lat (float): Latitude of starting location.
            lon (float): Longitude of starting location.
            heading (float): Heading (look direction).
            max_alt (float, optional): Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
            n_pts (int, optional): Number of GEO coordinate angular grid points (i.e. number of GLOW runs). Defaults to 200.
            n_bins (int, optional): Number of energy bins. Defaults to 100.
            n_threads (int, optional): Number of threads for parallel GLOW runs. Defaults to 24.
        """
        grobj = cls(time, lat, lon, heading, max_alt,
                    n_pts, n_bins, n_threads=n_threads)
        bds = grobj.run_no_precipitation()
        return bds

    def run_no_precipitation(self) -> xr.Dataset:
        """Run the GLOW model calculation to get the model output in GEO coordinates.

        Returns:
            xr.Dataset: GLOW model output in GEO coordinates.
        """
        if self._bds is not None:
            return self._bds
        # calculate the GLOW model for each lat-lon point determined in init()
        self._bds = self._calc_glow_noprecip()
        return self._bds  # return the calculated

    def transform_coord(self) -> xr.Dataset:
        """Run the coordinate transform to convert GLOW output from GEO to local coordinate system.

        Returns:
            xr.Dataset: GLOW output in (ZA, r) coordinates.
        """
        if self._bds is None:
            _ = self.run_no_precipitation()
        if self._iono is not None:
            return self._iono
        tt, rr = self._get_local_coords(
            self._bds.angle.values, self._bds.alt_km.values + EARTH_RADIUS)  # get local coords from geocentric coords
        self._rmin, self._rmax = 0, rr.max()  # nearest and farthest local pts
        # highest and lowest look angle (90 deg - za)
        self._tmin, self._tmax = 0, tt.max()
        self._nr_num = len(self._bds.alt_km.values) * \
            2  # resample to double density
        self._nt_num = len(self._bds.angle.values) * \
            2  # resample to double density
        self._altkm = altkm = self._bds.alt_km.values  # store the altkm
        self._theta = theta = self._bds.angle.values  # store the angles
        rmin, rmax = self._rmin, self._rmax  # local names
        tmin, tmax = self._tmin, self._tmax
        self._nr = nr = np.linspace(
            rmin, rmax, self._nr_num, endpoint=True)  # local r
        self._nt = nt = np.linspace(
            tmin, tmax, self._nt_num, endpoint=True)  # local look angle
        # get meshgrid of the R, T coord system from regular r, la grid
        self._ntt, self._nrr = self._get_global_coords(nt, nr)
        self._ntt = self._ntt.flatten()  # flatten T, works as _global_from_local LUT
        self._nrr = self._nrr.flatten()  # flatten R, works as _global_from_local LUT
        self._ntt = (self._ntt - self._theta.min()) / \
            (self._theta.max() - self._theta.min()) * \
            len(self._theta)  # calculate equivalent index (pixel coord) from original T grid
        self._nrr = (self._nrr - self._altkm.min() - EARTH_RADIUS) / \
            (self._altkm.max() - self._altkm.min()) * \
            len(self._altkm)  # calculate equivalent index (pixel coord) from original R (alt_km) grid
        data_vars = {}
        bds = self._bds
        coord_wavelength = bds.wavelength.values  # wl axis
        coord_state = bds.state.values  # state axis
        coord_energy = bds.energy.values  # egrid
        bds_attr = bds.attrs  # attrs
        single_keys = ['Tn',
                       'O',
                       'N2',
                       'O2',
                       'NO',
                       'NeIn',
                       'NeOut',
                       'ionrate',
                       'O+',
                       'O2+',
                       'NO+',
                       'N2D',
                       'pedersen',
                       'hall',
                       'Te',
                       'Ti']  # (angle, alt_km) vars
        state_keys = [
            'production',
            'loss',
            'excitedDensity'
        ]  # (angle, alt_km, state) vars
        # start = perf_counter_ns()
        # map all the single key types from (angle, alt_km) -> (la, r)
        for key in single_keys:
            inp = self._bds[key].values
            inp[np.where(np.isnan(inp))] = 0
            out = geometric_transform(inp, mapping=self._global_from_local, output_shape=(
                inp.shape[0]*2, inp.shape[1]*2))
            data_vars[key] = (('za', 'r'), out)
        # end = perf_counter_ns()
        # print('Single_key conversion: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        # dataset of (angle, alt_km) vars
        iono = xr.Dataset(data_vars=data_vars, coords={
                          'za': np.pi/2 - nt, 'r': nr})
        # end = perf_counter_ns()
        # print('Single_key dataset: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        ver = []
        # map all the wavelength data from (angle, alt_km, wavelength) -> (la, r, wavelength)
        for key in coord_wavelength:
            inp = bds['ver'].loc[dict(wavelength=key)].values
            inp[np.where(np.isnan(inp))] = 0
            out = geometric_transform(inp, mapping=self._global_from_local, output_shape=(
                inp.shape[0]*2, inp.shape[1]*2))
            ver.append(out.T)
        # end = perf_counter_ns()
        # print('VER eval: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        ver = np.asarray(ver).T
        ver = xr.DataArray(
            ver,
            coords={'za': np.pi/2 - nt, 'r': nr,
                    'wavelength': coord_wavelength},
            dims=['za', 'r', 'wavelength'],
            name='ver'
        )  # create wl dataset
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
                    out = geometric_transform(inp, mapping=self._global_from_local, output_shape=(
                        inp.shape[0]*2, inp.shape[1]*2))
                    return out.T
                res = list(map(convert_state_stuff, coord_state))
                res = np.asarray(res).T
                d[key] = (('za', 'r', 'state'), res)
            # end = perf_counter_ns()
            # print('Prod_Loss Eval: %.3f us'%((end - start)*1e-3))
            # start = perf_counter_ns()
            prodloss = xr.Dataset(
                data_vars=d,
                coords={'za': np.pi/2 - nt, 'r': nr, 'state': coord_state}
            )  # calculate (angle, alt_km, state) -> (la, r, state) dataset
        else:
            prodloss = xr.Dataset()
        # end = perf_counter_ns()
        # print('Prod_Loss DS: %.3f us'%((end - start)*1e-3))
        ## EGrid conversion (angle, energy) -> (r, energy) ##
        # EGrid is avaliable really at (angle, alt_km = 0, energy)
        # So we get local coords for (angle, R=R0)
        # we discard the angle information because it is meaningless, EGrid is spatial
        # start = perf_counter_ns()
        _rr, _ = self._get_local_coords(
            bds.angle.values, np.ones(bds.angle.values.shape)*EARTH_RADIUS)
        _rr = rr[:, 0]  # spatial EGrid
        d = []
        for en in coord_energy:  # for each energy
            inp = bds['precip'].loc[dict(energy=en)].values
            # interpolate to appropriate energy grid
            out = np.interp(nr, _rr, inp)
            d.append(out)
        d = np.asarray(d).T
        precip = xr.Dataset({'precip': (('r', 'energy'), d)}, coords={
                            'r': nr, 'energy': coord_energy})
        # end = perf_counter_ns()
        # print('Precip interp and ds: %.3f us'%((end - start)*1e-3))

        # start = perf_counter_ns()
        iono = xr.merge((iono, ver, prodloss, precip))  # merge all datasets
        iono.attrs.update(bds_attr)  # copy original attrs
        # end = perf_counter_ns()
        # print('Merging: %.3f us'%((end - start)*1e-3))
        self._iono = iono
        return iono

    # get global coord index from local coord index, implemented as LUT
    def _global_from_local(self, pt: tuple(int, int)) -> tuple(float, float):
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

    def _get_global_coords(self, t: np.ndarray | float, r: np.ndarray | float, r0: float = EARTH_RADIUS, meshgrid: bool = True) -> tuple(np.ndarray, np.ndarray):
        if isinstance(r, np.ndarray) and (t, np.ndarray):  # if array
            if r.ndim != t.ndim:  # if dims don't match get out
                raise RuntimeError
            if r.ndim == 1 and meshgrid:
                _r, _t = np.meshgrid(r, t)
            elif r.ndim == 1 and not meshgrid:
                _r, _t = r, t
            else:
                _r, _t = r.copy(), t.copy()  # already a meshgrid?
                r = _r[0]
                t = _t[:, 0]
        elif isinstance(r, float) and (t, float):  # floats
            _r = np.atleast_1d(r)
            _t = np.atleast_1d(t)
        else:
            raise RuntimeError
        _t = np.pi/2 - _t
        rr = np.sqrt((_r*np.cos(_t) + r0)**2 +
                     (_r*np.sin(_t))**2)  # r, la to R, T
        tt = np.arctan2(_r*np.sin(_t), _r*np.cos(_t) + r0)
        return tt, rr

    def _get_local_coords(self, t: np.ndarray | float, r: np.ndarray | float, r0: float = EARTH_RADIUS) -> tuple(np.ndarray, np.ndarray):
        if isinstance(r, np.ndarray) and (t, np.ndarray):
            if r.ndim != t.ndim:
                raise RuntimeError
            if r.ndim == 1:
                _r, _t = np.meshgrid(r, t)
            else:
                _r, _t = r.copy(), t.copy()
                r = _r[0]
                t = _t[:, 0]
        elif isinstance(r, float) and (t, float):
            _r = np.atleast_1d(r)
            _t = np.atleast_1d(t)
        else:
            raise RuntimeError
        rr = np.sqrt((_r*np.cos(_t) - r0)**2 +
                     (_r*np.sin(_t))**2)  # R, T to r, la
        tt = np.pi/2 - np.arctan2(_r*np.sin(_t), _r*np.cos(_t) - r0)
        return tt, rr

    def _get_angle(self) -> np.ndarray:  # get haversine angles between two lat, lon coords
        angs = []
        for pt in self._locs:
            ang = haversine(self._locs[0], pt, unit=Unit.RADIANS)
            angs.append(ang)
        return np.asarray(angs)

    # calculate glow model for one location
    def _calc_glow_single_noprecip(self, index):
        d = self._locs[index]
        return glow.no_precipitation(self._time, d[0], d[1], self._nbins)

    def _calc_glow_noprecip(self) -> xr.Dataset:  # run glow model calculation
        self._dss = MAP_FCN(self._calc_glow_single_noprecip, range(
            len(self._locs)), max_workers=self._nthr)
        # for dest in tqdm(self._locs):
        #     self._dss.append(glow.no_precipitation(time, dest[0], dest[1], self._nbins))
        bds: xr.Dataset = xr.concat(
            self._dss, pd.Index(self._angs, name='angle'))
        latlon = np.asarray(self._locs)
        bds = bds.assign_coords(lat=('angle', latlon[:, 0]))
        bds = bds.assign_coords(lon=('angle', latlon[:, 1]))
        return bds


# %%
if __name__ == '__main__':
    time = datetime(2022, 2, 15, 6, 0).astimezone(pytz.utc)
    print(time)
    lat, lon = 42.64981361744372, -71.31681056737486
    grobj = GLOWRaycast(time, 42.64981361744372, -71.31681056737486, 40, n_threads=6, n_pts=100)
    st = perf_counter_ns()
    bds = grobj.run_no_precipitation()
    end = perf_counter_ns()
    print('Time to generate:', (end - st)*1e-6, 'ms')
    st = perf_counter_ns()
    iono = grobj.transform_coord()
    end = perf_counter_ns()
    print('Time to convert:', (end - st)*1e-6, 'ms')

# %%
