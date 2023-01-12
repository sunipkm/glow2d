# %% Imports
from __future__ import annotations
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import ncarglow as glow
import geomagindices as gi
from datetime import datetime
import pytz
from geopy import Point
from geopy.distance import GreatCircleDistance, EARTH_RADIUS
from haversine import haversine, Unit
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from scipy.ndimage import geometric_transform
from time import perf_counter_ns

# %%
class GLOWRaycast:
    def __init__(self, time: datetime, lat: float, lon: float, heading: float, max_alt: float = 1000, n_pts: int = 200, n_bins: int = 100, n_threads: int = 12, full_circ: bool = False):
        self._pt = Point(lat, lon)
        self._time = time
        max_d = 6400 * np.pi if full_circ else EARTH_RADIUS * np.arccos(EARTH_RADIUS / (EARTH_RADIUS + max_alt)) # 6400 * np.pi
        self._xyrange = (EARTH_RADIUS, max_d, EARTH_RADIUS + max_alt)
        distpts = np.linspace(0, max_d, n_pts, endpoint=True)
        self._locs = []
        self._nbins = n_bins
        self._nthr = n_threads
        for d in distpts:
            dest = GreatCircleDistance(kilometers=d).destination(self._pt, heading)
            self._locs.append((dest.latitude, dest.longitude))
        if full_circ:
            npt = self._locs[-1]
            for d in distpts:
                dest = GreatCircleDistance(kilometers=d).destination(npt, heading)
                self._locs.append((dest.latitude, dest.longitude))
        self._angs = self._get_angle()
        plt.plot(self._angs)
        plt.show()
        if full_circ:
            plt.plot(self._angs)
            self._angs[len(self._angs) // 2:] = 2*np.pi - self._angs[len(self._angs) // 2:]
            plt.plot(self._angs)
            plt.show()
    
    def run(self):
        self._bds = self._calc_glow()
        return self._bds

    def transform_coord(self):
        rr, tt = self._get_local_coords(self._bds.alt_km.values + EARTH_RADIUS, self._bds.angle.values)
        self._rmin, self._rmax = rr.min(), rr.max()
        self._tmin, self._tmax = 0, tt.max()
        self._nr_num = len(self._bds.alt_km.values)*2
        self._nt_num = len(self._bds.angle.values)*2
        self._altkm = altkm = self._bds.alt_km.values
        self._theta = theta = self._bds.angle.values
        rmin, rmax = rr.min(), rr.max()
        tmin, tmax = 0, tt.max()
        self._nr = nr = np.linspace(rmin, rmax, self._nr_num)
        self._nt = nt = np.linspace(tmin, tmax, self._nt_num)
        inp = self._bds.Te.values
        inp[np.where(np.isnan(inp))] = 0
        out = geometric_transform(inp, mapping=self._global_from_local, output_shape=(inp.shape[0]*2, inp.shape[1]*2))
        return (inp, rr, tt), (out, nr, nt)

    def _global_from_local(self, pt):
        # if not self.firstrun % 10000:
        #     print('Input:', pt)
        tl, rl = pt # pixel coord
        # rl = (rl * (self._rmax - self._rmin) / self._nr_num) + self._rmin # real coord
        # tl = (tl * (self._tmax - self._tmin) / self._nt_num) + self._tmin  
        rl = self._nr[rl]
        tl = self._nt[tl]  
        r, t = self._get_global_coords(rl, tl)
        # if not self.firstrun % 10000:
        #     print((rl, tl), '->', (r, t), ':', (self._altkm.min(), self._altkm.max()))
        r = (r - self._altkm.min() - EARTH_RADIUS) / (self._altkm.max() - self._altkm.min()) * len(self._altkm)
        t = (t - self._theta.min()) / (self._theta.max() - self._theta.min()) * len(self._theta)
        # if not self.firstrun % 10000:
        #     print((float(r), float(t)))
        return (float(t), float(r))

    def _get_global_coords(self, r: np.ndarray | float, t: np.ndarray | float, r0: float = EARTH_RADIUS, meshgrid: bool = True) -> tuple(np.ndarray, np.ndarray):
        if isinstance(r, np.ndarray) and (t, np.ndarray):
            if r.ndim != t.ndim:
                raise RuntimeError
            if r.ndim == 1 and meshgrid:
                _r, _t = np.meshgrid(r, t)
            elif r.ndim == 1 and not meshgrid: _r, _t = r, t
            else:
                _r, _t = r.copy(), t.copy()
                r = _r[0]
                t = _t[:, 0]
        elif isinstance(r, float) and (t, float):
            _r = np.atleast_1d(r)
            _t = np.atleast_1d(t)
        else:
            raise RuntimeError
        _t = np.pi/2 - _t
        rr = np.sqrt((_r*np.cos(_t) + r0)**2 + (_r*np.sin(_t))**2)
        tt = np.arctan2(_r*np.sin(_t), _r*np.cos(_t) + r0)
        return rr, tt 

    def _get_local_coords(self, r: np.ndarray | float, t: np.ndarray | float, r0: float = EARTH_RADIUS) -> tuple:
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
        rr = np.sqrt((_r*np.cos(_t) - r0)**2 + (_r*np.sin(_t))**2)
        tt = np.pi/2 - np.arctan2(_r*np.sin(_t), _r*np.cos(_t) - r0)
        return rr, tt 
    
    def _get_angle(self):
        angs = []
        for pt in self._locs:
            ang = haversine(self._locs[0], pt, unit=Unit.RADIANS)
            angs.append(ang)
        return np.asarray(angs)

    def _calc_glow_single(self, index):
        d = self._locs[index]
        return glow.no_precipitation(self._time, d[0], d[1], self._nbins)

    def _calc_glow(self) -> xr.Dataset:
        self._dss = process_map(self._calc_glow_single, range(len(self._locs)), max_workers=self._nthr)
        # for dest in tqdm(self._locs):
        #     self._dss.append(glow.no_precipitation(time, dest[0], dest[1], self._nbins))
        bds: xr.Dataset = xr.concat(self._dss, pd.Index(self._angs, name='angle'))
        latlon = np.asarray(self._locs)
        bds = bds.assign_coords(lat=('angle', latlon[:, 0]))
        bds = bds.assign_coords(lon=('angle', latlon[:, 1]))
        return bds

# %%
time = datetime(2022, 2, 15, 6, 0).astimezone(pytz.utc)
print(time)
lat, lon = 42.64981361744372, -71.31681056737486
grobj = GLOWRaycast(time, 42.64981361744372, -71.31681056737486, 40)
bds = grobj.run()
# %%
tn = np.log10(bds.ver.loc[dict(wavelength='5577')].values)
alt = bds.alt_km.values
ang = bds.angle.values
# %%
import pylab as pl
ofst = 1000
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'))
r, t = np.meshgrid((alt + ofst) / ofst, ang)
print(r.shape, t.shape)
im = ax.contourf(t, r, tn, 100) #, extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
fig.colorbar(im)
earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
ax.add_artist(earth)
ax.set_thetamax(ang.max()*180/np.pi)
ax.set_ylim([0, (alt.max() + ofst) / ofst])
# ax.set_rscale('symlog')
# ax.set_rorigin(-1)
plt.show()
# %%
def get_local_coords(r: np.ndarray | float, t: np.ndarray | float, r0: float = EARTH_RADIUS) -> tuple(np.ndarray, np.ndarray):
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
    rr = np.sqrt((_r*np.cos(_t) - r0)**2 + (_r*np.sin(_t))**2)
    tt = np.pi/2 - np.arctan2(_r*np.sin(_t), _r*np.cos(_t) - r0)
    return rr, tt

r, t = np.meshgrid((alt + EARTH_RADIUS), ang)
rr, tt = get_local_coords(r, t)
# %%
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'))
tn = np.log10(bds.ver.loc[dict(wavelength='6300')].values)
r, t = rr, tt # np.meshgrid((alt + ofst) / ofst, ang)
print(r.shape, t.shape)
im = ax.contourf(t, r, tn, 100) #, extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
fig.colorbar(im)
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
# ax.set_thetamax(ang.max()*180/np.pi)
# ax.set_ylim([0, (alt.max() + ofst) / ofst])
# ax.set_rscale('symlog')
# ax.set_rorigin(-1)
plt.show()
# %%
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'))
ax.scatter(rr, tt)
plt.show()
# %%
plt.hist2d(rr.flatten(), np.pi/2 - tt.flatten())
# %%
gr1, gr2 = grobj.transform_coord()
# %%
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'))
tn = gr1[0]
r, t = gr1[1], gr1[2] # np.meshgrid((alt + ofst) / ofst, ang)
print(r.shape, t.shape)
im = ax.contourf(t, r, tn, 100) #, extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
fig.colorbar(im)
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
# ax.set_thetamax(ang.max()*180/np.pi)
# ax.set_ylim([0, (alt.max() + ofst) / ofst])
# ax.set_rscale('symlog')
# ax.set_rorigin(-1)
plt.show()
# %%
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'))
tn = gr2[0]
r, t = gr2[1], gr2[2] # np.meshgrid((alt + ofst) / ofst, ang)
print(r.shape, t.shape)
r, t = np.meshgrid(r, t)
im = ax.contourf(t, r, tn, 100) #, extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
fig.colorbar(im)
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
# ax.set_thetamax(ang.max()*180/np.pi)
# ax.set_ylim([0, (alt.max() + ofst) / ofst])
# ax.set_rscale('symlog')
# ax.set_rorigin(-1)
plt.show()
# %%
def get_global_coords(r: np.ndarray | float, t: np.ndarray | float, r0: float = EARTH_RADIUS) -> tuple(np.ndarray, np.ndarray):
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
    _t = np.pi/2 - _t
    rr = np.sqrt((_r*np.cos(_t) + r0)**2 + (_r*np.sin(_t))**2)
    tt = np.arctan2(_r*np.sin(_t), _r*np.cos(_t) + r0)
    return rr, tt 

def get_local_coords(r: np.ndarray | float, t: np.ndarray | float, r0: float = EARTH_RADIUS) -> tuple:
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
    rr = np.sqrt((_r*np.cos(_t) - r0)**2 + (_r*np.sin(_t))**2)
    tt = np.pi/2 - np.arctan2(_r*np.sin(_t), _r*np.cos(_t) - r0)
    return rr, tt 
# %%
get_local_coords(60 + EARTH_RADIUS, np.pi/2)
# %%
get_global_coords(9052.49316134, -0.78071144)
# %%
6431.0089784 - EARTH_RADIUS
# %%
rr, tt = get_local_coords(bds.alt_km.values + EARTH_RADIUS, bds.angle.values)
# %%
get_global_coords(60.0, -0.24611431901614256)
# %%
tn = gr2[0]
r, t = gr2[1], gr2[2] # np.meshgrid((alt + ofst) / ofst, ang)

plt.imshow(tn)
# %%
r
# %%
rr, tt = get_local_coords(bds.alt_km.values + EARTH_RADIUS, bds.angle.values)
# %%
plt.hist2d(rr.flatten(), tt.flatten())
# %%
rrr, ttt = get_global_coords(rr, tt)
# %%
plt.hist2d(rrr.flatten(), ttt.flatten())
# %%
r, t = np.meshgrid(bds.alt_km.values + EARTH_RADIUS, bds.angle.values)
plt.hist2d(r.flatten(), t.flatten())
# %%
nr = np.linspace(rr.min(), rr.max(), 250)
nt = np.linspace(0, tt.max(), 200)
rrrr, tttt = get_global_coords(nr, nt)
# %%
plt.hist2d(rrrr.flatten() - EARTH_RADIUS, tttt.flatten())
# %%
gpts = np.asarray(grobj._geopts)
# %%
plt.hist2d(gpts[:, 0], gpts[:, 1])
# %%
x, y = bds.alt_km.values + EARTH_RADIUS, bds.angle.values
hpts = np.zeros(gpts.shape)
hpts[:, 0] = (gpts[:, 0] - x.min() - EARTH_RADIUS) / (x.max() - x.min())
hpts[:, 1] = (t - y.min()) / (y.max() - y.min())
# %%
fig, ax = plt.subplots(dpi=300) #, subplot_kw=dict(projection='polar'))
tn = gr2[0]
r, t = gr2[1], gr2[2] # np.meshgrid((alt + ofst) / ofst, ang)
print(r.shape, t.shape)
r, t = np.meshgrid(r, t)
im = ax.imshow(tn.transpose()) # ax.contourf(t, r, tn, 100) #, extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
fig.colorbar(im)
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
# ax.set_thetamax(ang.max()*180/np.pi)
# ax.set_ylim([0, (alt.max() + ofst) / ofst])
# ax.set_rscale('symlog')
# ax.set_rorigin(-1)
plt.show()
# %%
