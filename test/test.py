# %% Imports
from __future__ import annotations
import pylab as pl
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import EARTH_RADIUS
from datetime import datetime
import pytz
from time import perf_counter_ns
from glow2d import glow2d_geo, glow2d_polar

# %%
time = datetime(2022, 2, 15, 6, 0).astimezone(pytz.utc)
print(time)
lat, lon = 42.64981361744372, -71.31681056737486
grobj = glow2d_geo(time, 42.64981361744372, -71.31681056737486, 40, n_pts = 100)
st = perf_counter_ns()
bds = grobj.run_no_precipitation()
end = perf_counter_ns()
print('Time to generate:', (end - st)*1e-6, 'ms')
st = perf_counter_ns()
grobj = glow2d_polar(bds, resamp=2)
iono = grobj.transform_coord()
end = perf_counter_ns()
print('Time to convert:', (end - st)*1e-6, 'ms')
# %%
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
import matplotlib
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FormatStrFormatter
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pylab as pl
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)
# %% 5577
ofst = 1000
scale = 1000
fig = plt.figure(figsize=(4.8, 3.8), dpi=300, constrained_layout=True)
gspec = GridSpec(2, 1, hspace=0.02, height_ratios=[1, 100], figure=fig)
ax = fig.add_subplot(gspec[1, 0], projection='polar')
cax = fig.add_subplot(gspec[0, 0])
# fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300, subplot_kw=dict(projection='polar'), constrained_layout=True, squeeze=True)
# fig.subplots_adjust(right=0.8)
# cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('top', size='5%', pad=0.05)
tn = (bds.ver.loc[dict(wavelength='5577')].values)
alt = bds.alt_km.values
ang = bds.angle.values
r, t = (alt + ofst) / scale, ang  # np.meshgrid((alt + ofst), ang)
print(r.shape, t.shape)
# , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
im = ax.contourf(t, r, tn.T, 100, cmap='gist_ncar_r')
cbar = fig.colorbar(im, cax=cax, shrink=0.6, orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
cbar.set_label(r'Volume Emission Rate ($%s$)'%(bds.ver.attrs['units']), fontsize=8)
earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
ax.add_artist(earth)
ax.set_thetamax(ang.max()*180/np.pi)
ax.set_ylim([0, (600 / scale) + 1])
locs = ax.get_yticks()


def get_loc_labels(locs, ofst, scale):
    locs = np.asarray(locs)
    locs = locs[np.where(locs > 1.0)]
    labels = ['O', r'R$_0$']
    for loc in locs:
        labels.append('%.0f' % (loc*scale - ofst))
    locs = np.concatenate((np.asarray([0, 1]), locs.copy()))
    labels = labels
    return locs, labels


locs, labels = get_loc_labels(locs, ofst, scale)
ax.set_yticks(locs)
ax.set_yticklabels(labels)

# label_position=ax.get_rlabel_position()
ax.text(np.radians(-12), ax.get_rmax()/2, 'Distance from Earth center (km)',
        rotation=0, ha='center', va='center')
ax.set_position([0.1, -0.45, 0.8, 2])
fig.suptitle('GLOW Model Brightness of 557.7 nm feature')
# ax.set_rscale('ofst_r_scale')
# ax.set_rscale('symlog')
# ax.set_rorigin(-1)
plt.savefig('test_geo_5577.pdf')
plt.show()
# %%
r, t = np.meshgrid((alt + EARTH_RADIUS), ang)
tt, rr = grobj.get_local_coords(t, r)
# %%
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'))
tn = (bds.ver.loc[dict(wavelength='5577')].values)
r, t = rr, np.pi / 2 - tt  # np.meshgrid((alt + ofst) / ofst, ang)
print(r.shape, t.shape)
# , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
im = ax.contourf(t, r, np.log10(tn), 100)
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
ax.set_thetamax(90)
ax.set_ylim(alt.min(), alt.max())
# ax.set_rscale('symlog')
ax.set_rorigin(-alt.min())
plt.show()
# %%
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(6.4, 4.8))
tn = (bds.ver.loc[dict(wavelength='5577')].values).copy()
vidx = np.where(t < 0)
tn[vidx] = 0
print(tn.sum())
# np.meshgrid((alt + ofst) / ofst, ang)
# , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
im = ax.contourf(t, r, tn, 100) #, cmap='gist_ncar_r')
cbar = fig.colorbar(im, shrink=0.6)
cbar.set_label(r'Volume Emission Rate ($%s$)'%(bds.ver.attrs['units']), fontsize=10)
cbar.ax.tick_params(labelsize=8)
ax.set_thetamax(90)
ax.text(np.radians(-12), ax.get_rmax()/2, 'Distance from observation location (km)',
        rotation=0, ha='center', va='center')
fig.suptitle('GLOW Model Brightness of 557.7 nm feature (local coordinates)')
ax.fill_between(np.deg2rad([12, 69]), 0, 10000, alpha=0.3, color='b')
ax.plot(np.deg2rad([12, 12]), [0, 10000], lw=0.5, color='k', ls='--')
ax.plot(np.deg2rad([69, 69]), [0, 10000], lw=0.5, color='k', ls='--')
ax.text(np.deg2rad(37), 1600, 'HiT&MIS View Cone', fontsize=10, color='w', rotation=360-45)
ax.tick_params(labelsize=10)
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
# ax.set_thetamax(ang.max()*180/np.pi)
ax.set_ylim(r.min(), r.max())
# ax.set_rscale('symlog')
ax.set_rorigin(-r.min())
plt.savefig('test_loc_5577.pdf')
plt.show()
# %%
res = np.histogram2d(rr.flatten(), tt.flatten(), range=([rr.min(), rr.max()], [0, np.pi / 2]))
# %%
# plt.imshow(tn[::20, ::20], extent=[res[0].min(), res[0].max(), res[1].min(), res[1].max()])
# plt.show()# %%
from scipy.signal import savgol_filter
from skimage.transform import resize

gd = resize(res[0], iono.ver.loc[dict(wavelength='5577')].values.shape, mode='edge')
gd *= res[0].sum() / gd.sum()
plt.imshow(gd)
plt.colorbar()
plt.show()
# %%

gd2 = savgol_filter(gd, 51, 5, mode='nearest')

print(gd.sum(), gd2.sum())
# %%
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(6.4, 4.8))
tn = iono.ver.loc[dict(wavelength='5577')].values
print('total:', tn.sum())
# np.meshgrid((alt + ofst) / ofst, ang)
r, t = iono.r.values, np.pi/2 - iono.za.values
print(r.shape, t.shape)
r, t = np.meshgrid(r, t)
# , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
im = ax.contourf(t, r, tn, 100) #, cmap='gist_ncar_r')
cbar = fig.colorbar(im, shrink=0.6)
cbar.set_label(r'Volume Emission Rate ($%s$)'%(iono.ver.attrs['units']), fontsize=10)
cbar.ax.tick_params(labelsize=8)
ax.set_thetamax(90)
ax.text(np.radians(-12), ax.get_rmax()/2, 'Distance from observation location (km)',
        rotation=0, ha='center', va='center')
fig.suptitle('GLOW Model Brightness of 557.7 nm feature (local coordinates, interpolated)')
ax.fill_between(np.deg2rad([12, 69]), 0, 10000, alpha=0.3, color='b')
ax.plot(np.deg2rad([12, 12]), [0, 10000], lw=0.5, color='k', ls='--')
ax.plot(np.deg2rad([69, 69]), [0, 10000], lw=0.5, color='k', ls='--')
ax.text(np.deg2rad(37), 1600, 'HiT&MIS View Cone', fontsize=10, color='w', rotation=360-45)
ax.tick_params(labelsize=10)
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
# ax.set_thetamax(ang.max()*180/np.pi)
ax.set_ylim(r.min(), r.max())
# ax.set_rscale('symlog')
ax.set_rorigin(-r.min())
plt.savefig('test_loc_5577_uniform.pdf')
plt.show()
# %%
from matplotlib import ticker
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(6.4, 4.8))
# np.meshgrid((alt + ofst) / ofst, ang)
r, t = iono.r.values, iono.za.values
print(r.shape, t.shape)
r, t = np.meshgrid(r, t)
tt, rr = grobj.get_global_coords(t, r)
gd2 = grobj.get_jacobian_glob2loc_glob(rr, tt)
t = np.pi / 2 - t
# , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
im = ax.contourf(t, r, gd2, 100, cmap='gist_ncar_r')
cbar = fig.colorbar(im, shrink=0.6)
cbar.set_label('Number Density', fontsize=10)
cbar.ax.tick_params(labelsize=8)
ax.set_thetamax(90)
ax.text(np.radians(-12), ax.get_rmax()/2, 'Distance from observation location (km)',
        rotation=0, ha='center', va='center')
fig.suptitle('Distribution of GEO coordinate points in local coordinates')
ax.fill_between(np.deg2rad([12, 69]), 0, 10000, alpha=0.3, color='b')
ax.plot(np.deg2rad([12, 12]), [0, 10000], lw=0.5, color='k', ls='--')
ax.plot(np.deg2rad([69, 69]), [0, 10000], lw=0.5, color='k', ls='--')
ax.text(np.deg2rad(37), 1600, 'HiT&MIS View Cone', fontsize=10, color='w', rotation=360-45)
ax.tick_params(labelsize=10)
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
# ax.set_thetamax(ang.max()*180/np.pi)
ax.set_ylim(r.min(), r.max())
# ax.set_rscale('symlog')
ax.set_rorigin(-60)
plt.savefig('test_loc_geo_distrib.pdf')
plt.show()
# %%
iono.za.values
# %%
iono.Tn.loc[dict(za=0)].plot()
bds.Tn.loc[dict(angle=0)].plot()
plt.xlim(60, 200)
# %%
# %%
bds.Tn.loc[dict(angle=0)].alt_km.values
# %%
iono.Tn.loc[dict(za=0)].r.values
# %%
glow2d_polar.get_local_coords(0, EARTH_RADIUS + 60)
# %%
gt, gr = glow2d_polar.get_global_coords(np.asarray([0]*len(iono.Tn.loc[dict(za=0)].r.values)), iono.Tn.loc[dict(za=0)].r.values, meshgrid=False)
# %%
plt.plot(iono.ver.loc[dict(za=0, wavelength='5577')].r, iono.ver.loc[dict(za=0, wavelength='5577')].values, label='Map')
plt.plot(bds.Te.loc[dict(angle=0)].alt_km.values, bds.ver.loc[dict(angle=0, wavelength='5577')].values, label='Orig')
plt.plot(gr - EARTH_RADIUS, np.interp(gr - EARTH_RADIUS, bds.Te.loc[dict(angle=0)].alt_km.values, bds.ver.loc[dict(angle=0, wavelength='5577')].values), label='Interp')
plt.xlim(60, 1000)
plt.legend()
# %%
plt.plot(iono.ver.loc[dict(za=0, wavelength='5577')].r, iono.ver.loc[dict(za=0, wavelength='5577')].values, label='Map')
plt.xlim(60, 1000)
# %%
from scipy.integrate import trapz

print(trapz(iono.ver.loc[dict(za=np.pi / 2, wavelength='5577')].values, iono.ver.loc[dict(za=np.pi / 2, wavelength='5577')].r))
print(trapz(bds.ver.loc[dict(angle=bds.angle.max(), wavelength='5577')].values, bds.ver.loc[dict(angle=bds.angle.max(), wavelength='5577')].alt_km + EARTH_RADIUS))
# %%
