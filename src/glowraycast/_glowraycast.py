# %% Imports
from __future__ import annotations
from typing import SupportsFloat as Numeric
import warnings
import xarray as xr
import numpy as np
from scipy.signal import savgol_filter
from skimage.transform import resize
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
from multiprocessing import cpu_count

MAP_FCN = process_map
if platform.system() == 'Darwin':
    MAP_FCN = thread_map

N_CPUS = cpu_count()

# %%
class GLOWRaycast:
    def __init__(self, time: datetime, lat: Numeric, lon: Numeric, heading: Numeric, max_alt: Numeric = 1000, n_pts: int = 50, n_bins: int = 100, *, with_prodloss: bool = False, n_threads: int = None, full_circ: bool = False, resamp: Numeric = 1.5):
        """Create a GLOWRaycast object.

        Args:
            time (datetime): Datetime of GLOW calculation.
            lat (Numeric): Latitude of starting location.
            lon (Numeric): Longitude of starting location.
            heading (Numeric): Heading (look direction).
            max_alt (Numeric, optional): Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
            n_pts (int, optional): Number of GEO coordinate angular grid points (i.e. number of GLOW runs), must be even and > 20. Defaults to 50.
            n_bins (int, optional): Number of energy bins. Defaults to 100.
            with_prodloss (bool, optional): Calculate production and loss parameters in local coordinates. Defaults to False.
            n_threads (int, optional): Number of threads for parallel GLOW runs. Set to None to use all system threads. Defaults to None.
            full_circ (bool, optional): For testing only, do not use. Defaults to False.
            resamp (Numeric, optional): Number of R and ZA points in local coordinate output. ``len(R) = len(alt_km) * resamp`` and ``len(ZA) = n_pts * resamp``. Must be > 0.5. Defaults to 1.5.

        Raises:
            ValueError: Number of position bins can not be odd.
            ValueError: Number of position bins can not be < 20.
            ValueError: Resampling can not be < 0.5.

        Warns:
            ResourceWarning: Number of threads requested is more than available system threads.
        """
        if n_pts % 2:
            raise ValueError('Number of position bins can not be odd.')
        if n_pts < 20:
            raise ValueError('Number of position bins can not be < 20.')
        if n_threads is None:
            n_threads = N_CPUS
        if n_threads > N_CPUS:
            warnings.warn('Number of requested threads (%d > %d)' % (n_threads, N_CPUS), ResourceWarning)
        if resamp < 0.5:
            raise ValueError('Resampling can not be < 0.5.')
        self._resamp = resamp
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

    def run_no_precipitation(self) -> xr.Dataset:
        """Run the GLOW model calculation to get the model output in GEO coordinates.

        Returns:
            xr.Dataset: GLOW model output in GEO coordinates.
        """
        if self._bds is not None:
            return self._bds
        # calculate the GLOW model for each lat-lon point determined in init()
        self._bds = self._calc_glow_noprecip()
        unit_desc_dict ={
            'angle': ('radians', 'Angle of location w.r.t. radius vector at origin (starting location)'),
            'lat': ('degree', 'Latitude of locations'),
            'lon': ('degree', 'Longitude of location')
        }
        _ = list(map(lambda x: self._bds[x].attrs.update(
        {'units': unit_desc_dict[x][0], 'description': unit_desc_dict[x][1]}), unit_desc_dict.keys()))
        return self._bds  # return the calculated

    @classmethod
    def no_precipitation(cls, time: datetime, lat: Numeric, lon: Numeric, heading: Numeric, max_alt: Numeric = 1000, n_pts: int = 50, n_bins: int = 100, *, with_prodloss: bool = False, n_threads: int = None, full_output: bool = False, resamp: Numeric = 1.5) -> xr.Dataset | tuple(xr.Dataset, xr.Dataset):
        """Run GLOW model looking along heading from the current location and return the model output in
        (ZA, R) local coordinates where ZA is zenith angle in radians and R is distance in kilometers.

        Args:
            time (datetime): Datetime of GLOW calculation.
            lat (Numeric): Latitude of starting location.
            lon (Numeric): Longitude of starting location.
            heading (Numeric): Heading (look direction).
            max_alt (Numeric, optional): Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
            n_pts (int, optional): Number of GEO coordinate angular grid points (i.e. number of GLOW runs), must be even and > 20. Defaults to 50.
            n_bins (int, optional): Number of energy bins. Defaults to 100.
            with_prodloss (bool, optional): Calculate production and loss parameters in local coordinates. Defaults to False.
            n_threads (int, optional):  Number of threads for parallel GLOW runs. Set to None to use all system threads. Defaults to None.
            full_output (bool, optional): Returns only local coordinate GLOW output if False, and a tuple of local and GEO outputs if True. Defaults to False.
            resamp (Numeric, optional): Number of R and ZA points in local coordinate output. ``len(R) = len(alt_km) * resamp`` and ``len(ZA) = n_pts * resamp``. Must be > 0.5. Defaults to 1.5.

        Returns:
            iono (xarray.Dataset): Ionospheric parameters and brightnesses (with or without production and loss) in local coordinates.

            iono, bds (xarray.Dataset, xarray.Dataset): These values are returned only if ``full_output == True``.

            - Ionospheric parameters and brightnesses (with or without production and loss) in local coordinates.
            - Ionospheric parameters and brightnesses (with production and loss) in GEO coordinates.


        Raises:
            ValueError: Number of position bins can not be odd.
            ValueError: Number of position bins can not be < 20.
            ValueError: Resampling can not be < 0.5.

        Warns:
            ResourceWarning: Number of threads requested is more than available system threads.
        """
        grobj = cls(time, lat, lon, heading, max_alt, n_pts, n_bins,
                    n_threads=n_threads, with_prodloss=with_prodloss, resamp=resamp)
        bds = grobj.run_no_precipitation()
        iono = grobj.transform_coord()
        if not full_output:
            return iono
        else:
            return (iono, bds)

    @classmethod
    def no_precipitation_geo(cls, time: datetime, lat: Numeric, lon: Numeric, heading: Numeric, max_alt: Numeric = 1000, n_pts: int = 50, n_bins: int = 100, *, n_threads: int = None, resamp: Numeric = 1.5) -> xr.Dataset:
        """Run GLOW model looking along heading from the current location and return the model output in
        (T, R) geocentric coordinates where T is angle in radians from the current location along the great circle
        following current heading, and R is altitude in kilometers.

        Args:
            time (datetime): Datetime of GLOW calculation.
            lat (Numeric): Latitude of starting location.
            lon (Numeric): Longitude of starting location.
            heading (Numeric): Heading (look direction).
            max_alt (Numeric, optional): Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
            n_pts (int, optional): Number of GEO coordinate angular grid points (i.e. number of GLOW runs). Defaults to 50.
            n_bins (int, optional): Number of energy bins. Defaults to 100.
            n_threads (int, optional):  Number of threads for parallel GLOW runs. Set to None to use all system threads. Defaults to None.
            resamp (Numeric, optional): Number of R and ZA points in local coordinate output. ``len(R) = len(alt_km) * resamp`` and ``len(ZA) = n_pts * resamp``. Must be > 0.5. Defaults to 1.5.


        Returns:
            bds (xarray.Dataset): Ionospheric parameters and brightnesses (with production and loss) in GEO coordinates.

        Raises:
            ValueError: Number of position bins can not be odd.
            ValueError: Number of position bins can not be < 20.
            ValueError: Resampling can not be < 0.5.

        Warns:
            ResourceWarning: Number of threads requested is more than available system threads.
        """
        grobj = cls(time, lat, lon, heading, max_alt,
                    n_pts, n_bins, n_threads=n_threads, resamp=resamp)
        bds = grobj.run_no_precipitation()
        return bds

    def transform_coord(self) -> xr.Dataset:
        """Run the coordinate transform to convert GLOW output from GEO to local coordinate system.

        Returns:
            xr.Dataset: GLOW output in (ZA, r) coordinates.
        """
        if self._bds is None:
            _ = self.run_no_precipitation()
        if self._iono is not None:
            return self._iono
        tt, rr = self.get_local_coords(
            self._bds.angle.values, self._bds.alt_km.values + EARTH_RADIUS)  # get local coords from geocentric coords

        self._rmin, self._rmax = rr.min(), rr.max()  # nearest and farthest local pts
        # highest and lowest look angle (90 deg - za)
        self._tmin, self._tmax = 0, tt.max()
        self._nr_num = int(len(self._bds.alt_km.values) * self._resamp)  # resample to half density
        self._nt_num = int(len(self._bds.angle.values) * self._resamp)   # resample to half density
        outp_shape = (self._nt_num, self._nr_num)

        ttidx = np.where(tt < 0) # angle below horizon (LA < 0)
        res = np.histogram2d(rr.flatten(), tt.flatten(), range=([rr.min(), rr.max()], [0, tt.max()])) # get distribution of global -> local points in local grid
        gd = resize(res[0], outp_shape, mode='edge') # remap to right size
        gd *= res[0].sum() / gd.sum() # conserve sum of points
        window_length = int(25 * self._resamp) # smoothing window
        window_length = window_length if window_length % 2 else window_length + 1 # must be odd
        gd = savgol_filter(gd, window_length=window_length, polyorder=5, mode='nearest') # smooth the distribution

        self._altkm = altkm = self._bds.alt_km.values  # store the altkm
        self._theta = theta = self._bds.angle.values  # store the angles
        rmin, rmax = self._rmin, self._rmax  # local names
        tmin, tmax = self._tmin, self._tmax
        self._nr = nr = np.linspace(
            rmin, rmax, self._nr_num, endpoint=True)  # local r
        self._nt = nt = np.linspace(
            tmin, tmax, self._nt_num, endpoint=True)  # local look angle
        # get meshgrid of the R, T coord system from regular r, la grid
        self._ntt, self._nrr = self.get_global_coords(nt, nr)
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
            inp = self._bds[key].values.copy()
            inp[np.where(np.isnan(inp))] = 0
            out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape)
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
            inp = bds['ver'].loc[dict(wavelength=key)].values.copy()
            inp[np.where(np.isnan(inp))] = 0
            out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape, mode='nearest') * gd # scaled by point distribution because flux is conserved, not brightness
            inp[ttidx] = 0
            inpsum = inp.sum() # sum of input for valid angles
            outpsum = out.sum() # sum of output 
            out = out * (inpsum / outpsum) # scale the sum to conserve total flux
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
                    inp = bds[key].loc[dict(state=st)].values.copy()
                    inp[np.where(np.isnan(inp))] = 0
                    out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape)
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
        _rr, _ = self.get_local_coords(
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

        unit_dict = {'Tn': 'K',
                 'O': 'cm^-3',
                 'N2': 'cm^-3',
                 'O2': 'cm^-3',
                 'NO': 'cm^-3',
                 'NeIn': 'cm^-3',
                 'NeOut': 'cm^-3',
                 'ionrate': 'cm^-3',
                 'O+': 'cm^-3',
                 'O2+': 'cm^-3',
                 'NO+': 'cm^-3',
                 'N2D': 'cm^-3',
                 'pedersen': 'S m^-1',
                 'hall': 'S m^-1',
                 'Te': 'K',
                 'Ti': 'K',
                 'ver': 'R',
                 'wavelength': 'angstrom',
                 'energy': 'eV'
                 }

        description_dict = {'Tn': 'Neutral temperature',
                 'O': 'Number density',
                 'N2': 'Number density',
                 'O2': 'Number density',
                 'NO': 'Number density',
                 'NeIn': 'Number density',
                 'NeOut': 'Number density',
                 'ionrate': 'Number density',
                 'O+': 'Number density',
                 'O2+': 'Number density',
                 'NO+': 'Number density',
                 'N2D': 'Number density',
                 'pedersen': 'Pedersen conductivity',
                 'hall': 'Hall conductivity',
                 'Te': 'Electron temperature',
                 'Ti': 'Ion temperature',
                 'ver': 'Volume (column) photon emission rate',
                 'wavelength': 'Emission wavelength',
                 'energy': 'Precipitation energy'
                 }
        if self._wprodloss:
            unit_dict['production'] = 'cm^-3 s^-1'
            unit_dict['loss'] = 's^-1'
            unit_dict['excitedDensity'] = 'cm^-3'
            description_dict['production'] = 'Volume production rate',
            description_dict['loss'] = 'Loss rate',
            description_dict['excitedDensity'] = 'Excited/ionized grid density',
    
        _ = list(map(lambda x: iono[x].attrs.update(
        {'units': unit_dict[x], 'description': description_dict[x]}), unit_dict.keys()))
        unit_desc_dict ={
            'za': ('radians', 'Zenith angle'),
            'r': ('km', 'Radial distance in km')
        }
        _ = list(map(lambda x: iono[x].attrs.update(
        {'units': unit_desc_dict[x][0], 'description': unit_desc_dict[x][1]}), unit_desc_dict.keys()))
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

    @staticmethod
    def get_global_coords(t: np.ndarray | Numeric, r: np.ndarray | Numeric, r0: Numeric = EARTH_RADIUS, meshgrid: bool = True) -> tuple(np.ndarray, np.ndarray):
        """Get GEO coordinates from local coordinates.

        Args:
            t (np.ndarray | Numeric): Angles in radians.
            r (np.ndarray | Numeric): Distance in km.
            r0 (Numeric, optional): Distance to origin. Defaults to geopy.distance.EARTH_RADIUS.
            meshgrid (bool, optional): Optionally convert 1-D inputs to a meshgrid. Defaults to True.

        Raises:
            ValueError: ``r`` and ``t`` does not have the same dimensions
            TypeError: ``r`` and ``t`` are not ``numpy.ndarray``.

        Returns:
            (np.ndarray, np.ndarray): (angles, distances) in GEO coordinates.
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
        _t = np.pi/2 - _t
        rr = np.sqrt((_r*np.cos(_t) + r0)**2 +
                     (_r*np.sin(_t))**2)  # r, la to R, T
        tt = np.arctan2(_r*np.sin(_t), _r*np.cos(_t) + r0)
        return tt, rr

    @staticmethod
    def get_local_coords(t: np.ndarray | Numeric, r: np.ndarray | Numeric, r0: Numeric = EARTH_RADIUS) -> tuple(np.ndarray, np.ndarray):
        """Get local coordinates from GEO coordinates.

        Args:
            t (np.ndarray | Numeric): Angles in radians.
            r (np.ndarray | Numeric): Distance in km.
            r0 (Numeric, optional): Distance to origin. Defaults to geopy.distance.EARTH_RADIUS.

        Raises:
            ValueError: ``r`` and ``t`` does not have the same dimensions
            TypeError: ``r`` and ``t`` are not ``numpy.ndarray``.

        Returns:
            (np.ndarray, np.ndarray): (angles, distances) in local coordinates.
        """
        if isinstance(r, np.ndarray) and isinstance(t, np.ndarray):
            if r.ndim != t.ndim:
                raise ValueError('r and t does not have the same dimensions')
            if r.ndim == 1:
                _r, _t = np.meshgrid(r, t)
            else:
                _r, _t = r.copy(), t.copy()
                r = _r[0]
                t = _t[:, 0]
        elif isinstance(r, Numeric) and isinstance(t, Numeric):
            _r = np.atleast_1d(r)
            _t = np.atleast_1d(t)
        else:
            raise TypeError('r and t must be np.ndarray.')
        rr = np.sqrt((_r*np.cos(_t) - r0)**2 +
                     (_r*np.sin(_t))**2)  # R, T to r, la
        tt = np.pi/2 - np.arctan2(_r*np.sin(_t), _r*np.cos(_t) - r0)
        return tt, rr

class GLOWRaycastXY:
    def __init__(self, time: datetime, lat: Numeric, lon: Numeric, heading: Numeric, max_alt: Numeric = 1000, n_pts: int = 50, n_bins: int = 100, *, with_prodloss: bool = False, n_threads: int = None, full_circ: bool = False, resamp: Numeric = 1.5):
        """Create a GLOWRaycastXY object.

        Args:
            time (datetime): Datetime of GLOW calculation.
            lat (Numeric): Latitude of starting location.
            lon (Numeric): Longitude of starting location.
            heading (Numeric): Heading (look direction).
            max_alt (Numeric, optional): Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
            n_pts (int, optional): Number of GEO coordinate angular grid points (i.e. number of GLOW runs), must be even and > 20. Defaults to 50.
            n_bins (int, optional): Number of energy bins. Defaults to 100.
            with_prodloss (bool, optional): Calculate production and loss parameters in local coordinates. Defaults to False.
            n_threads (int, optional): Number of threads for parallel GLOW runs. Set to None to use all system threads. Defaults to None.
            full_circ (bool, optional): For testing only, do not use. Defaults to False.
            resamp (Numeric, optional): Number of X and Y points in local coordinate output. ``len(Y) = len(alt_km) * resamp`` and ``len(X) = n_pts * resamp``. Must be > 0.5. Defaults to 1.5.

        Raises:
            ValueError: Number of position bins can not be odd.
            ValueError: Number of position bins can not be < 20.
            ValueError: Resampling can not be < 0.5.

        Warns:
            ResourceWarning: Number of threads requested is more than available system threads.
        """
        if n_pts % 2:
            raise ValueError('Number of position bins can not be odd.')
        if n_pts < 20:
            raise ValueError('Number of position bins can not be < 20.')
        if n_threads is None:
            n_threads = N_CPUS
        if n_threads > N_CPUS:
            warnings.warn('Number of requested threads (%d > %d)' % (n_threads, N_CPUS), ResourceWarning)
        if resamp < 0.5:
            raise ValueError('Resampling can not be < 0.5.')
        self._resamp = resamp
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
    def no_precipitation(cls, time: datetime, lat: Numeric, lon: Numeric, heading: Numeric, max_alt: Numeric = 1000, n_pts: int = 50, n_bins: int = 100, *, with_prodloss: bool = False, n_threads: int = None, full_output: bool = False, resamp: Numeric = 1.5) -> xr.Dataset | tuple(xr.Dataset, xr.Dataset):
        """Run GLOW model looking along heading from the current location and return the model output in
        (X, Y) local coordinates where X and Y are conventional axes in kilometers.

        Args:
            time (datetime): Datetime of GLOW calculation.
            lat (Numeric): Latitude of starting location.
            lon (Numeric): Longitude of starting location.
            heading (Numeric): Heading (look direction).
            max_alt (Numeric, optional): Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
            n_pts (int, optional): Number of GEO coordinate angular grid points (i.e. number of GLOW runs), must be even and > 20. Defaults to 50.
            n_bins (int, optional): Number of energy bins. Defaults to 100.
            with_prodloss (bool, optional): Calculate production and loss parameters in local coordinates. Defaults to False.
            n_threads (int, optional):  Number of threads for parallel GLOW runs. Set to None to use all system threads. Defaults to None.
            full_output (bool, optional): Returns only local coordinate GLOW output if False, and a tuple of local and GEO outputs if True. Defaults to False.
            resamp (Numeric, optional): Number of X and Y points in local coordinate output. ``len(Y) = len(alt_km) * resamp`` and ``len(X) = n_pts * resamp``. Must be > 0.5. Defaults to 1.5.

        Returns:
            iono (xarray.Dataset): Ionospheric parameters and brightnesses (with or without production and loss) in local coordinates.

            iono, bds (xarray.Dataset, xarray.Dataset): These values are returned only if ``full_output == True``.

            - Ionospheric parameters and brightnesses (with or without production and loss) in local coordinates.
            - Ionospheric parameters and brightnesses (with production and loss) in GEO coordinates.


        Raises:
            ValueError: Number of position bins can not be odd.
            ValueError: Number of position bins can not be < 20.
            ValueError: Resampling can not be < 0.5.

        Warns:
            ResourceWarning: Number of threads requested is more than available system threads.
        """
        grobj = cls(time, lat, lon, heading, max_alt, n_pts, n_bins,
                    n_threads=n_threads, with_prodloss=with_prodloss, resamp=resamp)
        bds = grobj.run_no_precipitation()
        iono = grobj.transform_coord()
        if not full_output:
            return iono
        else:
            return (iono, bds)

    @classmethod
    def no_precipitation_geo(cls, time: datetime, lat: Numeric, lon: Numeric, heading: Numeric, max_alt: Numeric = 1000, n_pts: int = 50, n_bins: int = 100, *, with_prodloss: bool = False, n_threads: int = None, resamp: Numeric = 1.5) -> xr.Dataset:
        """Run GLOW model looking along heading from the current location and return the model output in
        (T, R) geocentric coordinates where T is angle in radians from the current location along the great circle
        following current heading, and R is altitude in kilometers.

        Args:
            time (datetime): Datetime of GLOW calculation.
            lat (Numeric): Latitude of starting location.
            lon (Numeric): Longitude of starting location.
            heading (Numeric): Heading (look direction).
            max_alt (Numeric, optional): Maximum altitude where intersection is considered (km). Defaults to 1000, i.e. exobase.
            n_pts (int, optional): Number of GEO coordinate angular grid points (i.e. number of GLOW runs). Defaults to 50.
            n_bins (int, optional): Number of energy bins. Defaults to 100.
            n_threads (int, optional):  Number of threads for parallel GLOW runs. Set to None to use all system threads. Defaults to None.
            resamp (Numeric, optional): Number of X and Y points in local coordinate output. ``len(Y) = len(alt_km) * resamp`` and ``len(X) = n_pts * resamp``. Must be > 0.5. Defaults to 1.5.


        Returns:
            bds (xarray.Dataset): Ionospheric parameters and brightnesses (with production and loss) in GEO coordinates.

        Raises:
            ValueError: Number of position bins can not be odd.
            ValueError: Number of position bins can not be < 20.
            ValueError: Resampling can not be < 0.5.

        Warns:
            ResourceWarning: Number of threads requested is more than available system threads.
        """
        grobj = cls(time, lat, lon, heading, max_alt,
                    n_pts, n_bins, n_threads=n_threads, resamp=resamp)
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
        unit_desc_dict ={
            'angle': ('radians', 'Angle of location w.r.t. radius vector at origin (starting location)'),
            'lat': ('degree', 'Latitude of locations'),
            'lon': ('degree', 'Longitude of location')
        }
        _ = list(map(lambda x: self._bds[x].attrs.update(
        {'units': unit_desc_dict[x][0], 'description': unit_desc_dict[x][1]}), unit_desc_dict.keys()))
        return self._bds  # return the calculated

    def transform_coord(self) -> xr.Dataset:
        """Run the coordinate transform to convert GLOW output from GEO to local coordinate system (X, Y).

        Returns:
            xr.Dataset: GLOW output in (X, Y) coordinates.
        """
        if self._bds is None:
            _ = self.run_no_precipitation()
        if self._iono is not None:
            return self._iono
        tt, rr = self.get_local_coords_xy(
            self._bds.angle.values, self._bds.alt_km.values + EARTH_RADIUS)  # get local coords from geocentric coords
        self._rmin, self._rmax = self._bds.alt_km.values.min(), self._bds.alt_km.values.max()  # nearest and farthest local pts
        # highest and lowest look angle (90 deg - za)
        self._tmin, self._tmax = tt.min(), tt.max() # 60 to 1000 km, y
        self._nr_num = int(len(self._bds.alt_km.values) * self._resamp)  # resample to half density
        self._nt_num = int(len(self._bds.angle.values) * self._resamp)   # resample to half density
        outp_shape = (self._nt_num, self._nr_num)

        ttidx = np.where(tt < 0) # angle below horizon (LA < 0)
        res = np.histogram2d(rr.flatten(), tt.flatten(), range=([self._rmin, self._rmax], [self._tmin, self._tmax])) # get distribution of global -> local points in local grid
        gd = resize(res[0], outp_shape, mode='edge') # remap to right size
        gd *= res[0].sum() / gd.sum() # conserve sum of points
        window_length = int(25 * self._resamp) # smoothing window
        window_length = window_length if window_length % 2 else window_length + 1 # must be odd
        gd = savgol_filter(gd, window_length=window_length, polyorder=5, mode='nearest') # smooth the distribution

        self._altkm = altkm = self._bds.alt_km.values  # store the altkm
        self._theta = theta = self._bds.angle.values  # store the angles
        rmin, rmax = self._rmin, self._rmax  # local names
        tmin, tmax = self._tmin, self._tmax
        self._nr = nr = np.linspace(
            rmin, rmax, self._nr_num, endpoint=True)  # local r
        self._nt = nt = np.linspace(
            tmin, tmax, self._nt_num, endpoint=True)  # local look angle
        # get meshgrid of the R, T coord system from regular r, la grid
        self._ntt, self._nrr = self.get_global_coords_xy(nt, nr)
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
            inp = self._bds[key].values.copy()
            inp[np.where(np.isnan(inp))] = 0
            out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape)
            data_vars[key] = (('za', 'r'), out)
        # end = perf_counter_ns()
        # print('Single_key conversion: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        # dataset of (angle, alt_km) vars
        iono = xr.Dataset(data_vars=data_vars, coords={
                          'x': nt, 'y': nr})
        # end = perf_counter_ns()
        # print('Single_key dataset: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        ver = []
        # map all the wavelength data from (angle, alt_km, wavelength) -> (la, r, wavelength)
        for key in coord_wavelength:
            inp = bds['ver'].loc[dict(wavelength=key)].values.copy()
            inp[np.where(np.isnan(inp))] = 0
            out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape, mode='nearest') * gd # scaled by point distribution because flux is conserved, not brightness
            inp[ttidx] = 0
            inpsum = inp.sum() # sum of input for valid angles
            outpsum = out.sum() # sum of output 
            out = out * (inpsum / outpsum) # scale the sum to conserve total flux
            ver.append(out.T)
        # end = perf_counter_ns()
        # print('VER eval: %.3f us'%((end - start)*1e-3))
        # start = perf_counter_ns()
        ver = np.asarray(ver).T
        ver = xr.DataArray(
            ver,
            coords={'x': nt, 'y': nr,
                    'wavelength': coord_wavelength},
            dims=['x', 'y', 'wavelength'],
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
                    inp = bds[key].loc[dict(state=st)].values.copy()
                    inp[np.where(np.isnan(inp))] = 0
                    out = geometric_transform(inp, mapping=self._global_from_local, output_shape=outp_shape)
                    return out.T
                res = list(map(convert_state_stuff, coord_state))
                res = np.asarray(res).T
                d[key] = (('za', 'r', 'state'), res)
            # end = perf_counter_ns()
            # print('Prod_Loss Eval: %.3f us'%((end - start)*1e-3))
            # start = perf_counter_ns()
            prodloss = xr.Dataset(
                data_vars=d,
                coords={'x': nt, 'y': nr, 'state': coord_state}
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
        _rr, _ = self.get_local_coords_xy(
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
                            'y': nr, 'energy': coord_energy})
        # end = perf_counter_ns()
        # print('Precip interp and ds: %.3f us'%((end - start)*1e-3))

        # start = perf_counter_ns()
        iono = xr.merge((iono, ver, prodloss, precip))  # merge all datasets
        iono.attrs.update(bds_attr)  # copy original attrs
        
        unit_dict = {'Tn': 'K',
                 'O': 'cm^-3',
                 'N2': 'cm^-3',
                 'O2': 'cm^-3',
                 'NO': 'cm^-3',
                 'NeIn': 'cm^-3',
                 'NeOut': 'cm^-3',
                 'ionrate': 'cm^-3',
                 'O+': 'cm^-3',
                 'O2+': 'cm^-3',
                 'NO+': 'cm^-3',
                 'N2D': 'cm^-3',
                 'pedersen': 'S m^-1',
                 'hall': 'S m^-1',
                 'Te': 'K',
                 'Ti': 'K',
                 'ver': 'R',
                 'wavelength': 'angstrom',
                 'energy': 'eV'
                 }

        description_dict = {'Tn': 'Neutral temperature',
                 'O': 'Number density',
                 'N2': 'Number density',
                 'O2': 'Number density',
                 'NO': 'Number density',
                 'NeIn': 'Number density',
                 'NeOut': 'Number density',
                 'ionrate': 'Number density',
                 'O+': 'Number density',
                 'O2+': 'Number density',
                 'NO+': 'Number density',
                 'N2D': 'Number density',
                 'pedersen': 'Pedersen conductivity',
                 'hall': 'Hall conductivity',
                 'Te': 'Electron temperature',
                 'Ti': 'Ion temperature',
                 'ver': 'Volume (column) photon emission rate',
                 'wavelength': 'Emission wavelength',
                 'energy': 'Precipitation energy'
                 }
        if self._wprodloss:
            unit_dict['production'] = 'cm^-3 s^-1'
            unit_dict['loss'] = 's^-1'
            unit_dict['excitedDensity'] = 'cm^-3'
            description_dict['production'] = 'Volume production rate',
            description_dict['loss'] = 'Loss rate',
            description_dict['excitedDensity'] = 'Excited/ionized grid density',
    
        _ = list(map(lambda x: iono[x].attrs.update(
        {'units': unit_dict[x], 'description': description_dict[x]}), unit_dict.keys()))
        unit_desc_dict ={
            'x': ('km', 'Distance in X'),
            'y': ('km', 'Distance in Y')
        }
        _ = list(map(lambda x: iono[x].attrs.update(
        {'units': unit_desc_dict[x][0], 'description': unit_desc_dict[x][1]}), unit_desc_dict.keys()))
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

    @staticmethod
    def get_global_coords_xy(x: np.ndarray | Numeric, y: np.ndarray | Numeric, r0: Numeric = EARTH_RADIUS, meshgrid: bool = True) -> tuple(np.ndarray, np.ndarray):
        """Get GEO coordinates from local coordinates.

        Args:
            x (np.ndarray | Numeric): X distance in km.
            y (np.ndarray | Numeric): Y distance in km.
            r0 (Numeric, optional): Distance to origin. Defaults to geopy.distance.EARTH_RADIUS.
            meshgrid (bool, optional): Optionally convert 1-D inputs to a meshgrid. Defaults to True.

        Raises:
            ValueError: ``x`` and ``y`` does not have the same dimensions
            TypeError: ``x`` and ``y`` are not ``numpy.ndarray``.

        Returns:
            (np.ndarray, np.ndarray): (angles, distances) in GEO coordinates.
        """
        if isinstance(y, np.ndarray) and isinstance(x, np.ndarray):  # if array
            if y.ndim != x.ndim:  # if dims don't match get out
                raise ValueError('x and y does not have the same dimensions')
            if y.ndim == 1 and meshgrid:
                _r, _t = np.meshgrid(y, x)
            elif y.ndim == 1 and not meshgrid:
                _r, _t = y, x
            else:
                _r, _t = y.copy(), x.copy()  # already a meshgrid?
                y = _r[0]
                x = _t[:, 0]
        elif isinstance(y, Numeric) and isinstance(x, Numeric):  # floats
            _r = np.atleast_1d(y)
            _t = np.atleast_1d(x)
        else:
            raise TypeError('x and y must be np.ndarray.')
        # _t = np.pi/2 - _t
        # rr = np.sqrt((_r*np.cos(_t) + r0)**2 +
        #              (_r*np.sin(_t))**2)  # r, la to R, T
        # tt = np.arctan2(_r*np.sin(_t), _r*np.cos(_t) + r0)
        rr = np.sqrt((_r + r0)**2 + _t**2)
        tt = np.arctan2(_t, _r + r0)
        return tt, rr

    @staticmethod
    def get_local_coords_xy(t: np.ndarray | Numeric, r: np.ndarray | Numeric, r0: Numeric = EARTH_RADIUS) -> tuple(np.ndarray, np.ndarray):
        """Get local coordinates from GEO coordinates.

        Args:
            t (np.ndarray | Numeric): Angles in radians.
            r (np.ndarray | Numeric): Distance in km.
            r0 (Numeric, optional): Distance to origin. Defaults to geopy.distance.EARTH_RADIUS.

        Raises:
            ValueError: ``r`` and ``t`` does not have the same dimensions
            TypeError: ``r`` and ``t`` are not ``numpy.ndarray``.

        Returns:
            (np.ndarray, np.ndarray): (X, Y) in local coordinates.
        """
        if isinstance(r, np.ndarray) and isinstance(t, np.ndarray):
            if r.ndim != t.ndim:
                raise ValueError('r and t does not have the same dimensions')
            if r.ndim == 1:
                _r, _t = np.meshgrid(r, t)
            else:
                _r, _t = r.copy(), t.copy()
                r = _r[0]
                t = _t[:, 0]
        elif isinstance(r, Numeric) and isinstance(t, Numeric):
            _r = np.atleast_1d(r)
            _t = np.atleast_1d(t)
        else:
            raise TypeError('r and t must be np.ndarray.')
        # rr = np.sqrt((_r*np.cos(_t) - r0)**2 +
        #              (_r*np.sin(_t))**2)  # R, T to r, la
        # tt = np.pi/2 - np.arctan2(_r*np.sin(_t), _r*np.cos(_t) - r0)
        rr = _r*np.cos(_t) - r0
        tt = _r*np.sin(_t)
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
    grobj = GLOWRaycastXY(time, 42.64981361744372, -71.31681056737486, 40, n_threads=6, n_pts=100, resamp=1)
    st = perf_counter_ns()
    bds = grobj.run_no_precipitation()
    end = perf_counter_ns()
    print('Time to generate:', (end - st)*1e-6, 'ms')
    st = perf_counter_ns()
    iono = grobj.transform_coord()
    end = perf_counter_ns()
    print('Time to convert:', (end - st)*1e-6, 'ms')
