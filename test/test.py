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
from glowraycast import GLOWRaycast

# %%
time = datetime(2022, 2, 15, 6, 0).astimezone(pytz.utc)
print(time)
lat, lon = 42.64981361744372, -71.31681056737486
grobj = GLOWRaycast(time, 42.64981361744372, -71.31681056737486, 40, n_pts = 100)
st = perf_counter_ns()
bds = grobj.run_no_precipitation()
end = perf_counter_ns()
print('Time to generate:', (end - st)*1e-6, 'ms')
st = perf_counter_ns()
iono = grobj.transform_coord()
end = perf_counter_ns()
print('Time to convert:', (end - st)*1e-6, 'ms')
# %%
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FormatStrFormatter
import matplotlib
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
import datetime as dt

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
im = ax.contourf(t, r, tn.T, 100)
cbar = fig.colorbar(im, cax=cax, shrink=0.6, orientation='horizontal')
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Brightness (R)', fontsize=8)
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
fig.suptitle('GLOW Model Brightness of 557.7 nm feature (GEO coordinates)')
# ax.set_rscale('ofst_r_scale')
# ax.set_rscale('symlog')
# ax.set_rorigin(-1)
plt.savefig('test_geo_5577.pdf')
plt.show()
# %%
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(6.4, 4.8))
tn = iono.ver.loc[dict(wavelength='5577')].values
# np.meshgrid((alt + ofst) / ofst, ang)
r, t = iono.r.values, np.pi/2 - iono.za.values
print(r.shape, t.shape)
r, t = np.meshgrid(r, t)
# , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
im = ax.contourf(t, r, tn, 100)
fig.colorbar(im)
ax.set_thetamax(90)
ax.text(np.radians(-12), ax.get_rmax()/2, 'Distance from observation location (km)',
        rotation=0, ha='center', va='center')
fig.suptitle('GLOW Model Brightness of 557.7 nm feature (local coordinates)')
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
# ax.set_thetamax(ang.max()*180/np.pi)
# ax.set_ylim([0, (alt.max() + ofst) / ofst])
# ax.set_rscale('symlog')
# ax.set_rorigin(-1)
plt.savefig('test_loc_5577.pdf')
plt.show()

# %%
