# %% Imports
from __future__ import annotations
from ast import Tuple
import sys
from typing import Iterable
import pylab as pl
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import EARTH_RADIUS
from datetime import datetime
import pytz
from time import perf_counter_ns
from glow2d import glow2d_geo, glow2d_polar, polar_model
# %%
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc, rcParams
import matplotlib
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, FormatStrFormatter
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pylab as pl
from dateutil.parser import parse
from tzlocal import get_localzone
rc('font', **{'family': 'serif', 'serif': ['Times']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# %%
def get_minmax(iono: xr.Dataset, feature: str = 'Tn', subfeature: dict = None, minPositive: bool = True)->tuple(float, float):
    if subfeature is None:
        val = iono[feature].values
    else:
        val = iono[feature].loc[subfeature].values
    if minPositive:
        minval = val[np.where(val > 0)].min()
    else:
        minval = val.min()
    return (minval, val.max())

def get_all_minmax(ionos: dict[str, xr.Dataset], feature: str = 'Tn', subfeature: dict = None, minPositive: bool = True)->tuple(float, float):
    minmax = []
    for _, iono in ionos.items():
        minmax.append(get_minmax(iono, feature, subfeature, minPositive))
    minmax = np.asarray(minmax).T
    return minmax[0].min(), minmax[1].max()

def get_all_minmax_list(inp: Iterable, minPositive: bool = True)->tuple(float, float):
    def get_minmax_list(item: Iterable, minPositive: bool = True):
        item = np.asarray(item)
        if minPositive:
            minval = item[np.where(item > 0)].min()
        else:
            minval = item.min()
        return (minval, item.max())
    minmax = []
    for item in inp:
        minmax.append(get_minmax_list(item, minPositive))
    minmax = np.asarray(minmax).T
    return minmax[0].min(), minmax[1].max()

# %%
tdict = {
    'dawn': datetime(2022, 3, 22, 6, 0).astimezone(pytz.utc),
    'noon': datetime(2022, 3, 22, 12, 0).astimezone(pytz.utc),
    'dusk': datetime(2022, 3, 22, 18, 0).astimezone(pytz.utc),
    'midnight': datetime(2022, 3, 22, 23, 59).astimezone(pytz.utc)
}

bdss = {}
ionos = {}

for file_suffix, time in tdict.items():
    lat, lon = 42.64981361744372, -71.31681056737486
    grobj = glow2d_geo(time, 42.64981361744372, -71.31681056737486, 40, n_pts = 100)
    st = perf_counter_ns()
    bds = grobj.run_model()
    end = perf_counter_ns()
    print(f'Time to generate : {(end - st)*1e-6: 8.6f} ms')
    st = perf_counter_ns()
    grobj = glow2d_polar(bds, resamp=2)
    iono = grobj.transform_coord()
    end = perf_counter_ns()
    print(f'Time to convert  : {(end - st)*1e-6: 8.6f} ms')
    bdss[file_suffix] = bds
    ionos[file_suffix] = iono

# %% 5577
def plot_geo(bds: xr.Dataset, wl: str, file_suffix: str, *, vmin: float = None, vmax: float = None, decimals: int = 0, num_levels: int = 1000) -> None:
    ofst = 1000
    scale = 1000
    fig = plt.figure(figsize=(4.8, 3.8), dpi=300, constrained_layout=True)
    gspec = GridSpec(2, 1, hspace=0.02, height_ratios=[1, 100], figure=fig)
    ax = fig.add_subplot(gspec[1, 0], projection='polar')
    cax = fig.add_subplot(gspec[0, 0])
    dtime = parse(bds.time).astimezone(get_localzone())
    day = dtime.strftime('%Y-%m-%d')
    time_of_day = dtime.strftime('%H:%M hrs')
    # fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300, subplot_kw=dict(projection='polar'), constrained_layout=True, squeeze=True)
    # fig.subplots_adjust(right=0.8)
    # cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('top', size='5%', pad=0.05)
    tn = (bds.ver.loc[dict(wavelength=wl)].values)
    alt = bds.alt_km.values
    ang = bds.angle.values
    r, t = (alt + ofst) / scale, ang  # np.meshgrid((alt + ofst), ang)
    print(r.shape, t.shape)
    # , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
    ticks = None
    levels = num_levels
    if vmin is not None and vmax is not None:
        levels = np.linspace(np.log10(vmin), np.log10(vmax), num_levels, endpoint=True).tolist()
        ticks = np.linspace(np.log10(vmin), np.log10(vmax), 10, endpoint=True)
        ticks = np.unique(np.round(ticks, decimals=decimals))
    im = ax.contourf(t, r, np.log10(tn.T), cmap='gist_ncar_r', levels=levels)
    cbar = fig.colorbar(im, cax=cax, shrink=0.6, orientation='horizontal', ticks=ticks)
    if ticks is not None: cbar.ax.set_xticklabels([r'$10^{%d}$'%(tval) for tval in ticks])
    cbar.ax.tick_params(labelsize=10)
    # cbar.formatter.set_useMathText(True)
    cbar.set_label(r'%s \AA{} VER ($%s$)'%(wl, bds.ver.attrs['units']), fontsize=12)
    earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
    ax.add_artist(earth)
    ax.set_thetamax(ang.max()*180/np.pi)
    
    ax.scatter(0, 1, s=40, marker=r'$\odot$', facecolors='none', edgecolors='blue', clip_on=False)
    ax.scatter(np.deg2rad(90), 0.272, s=40, marker=r'$\odot$', facecolors='none', edgecolors='blue', clip_on=False)
    ax.text(np.deg2rad(75), 0.27, 'Observer', fontdict={'size': 12})
    
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
            rotation=0, ha='center', va='center', fontdict={'size': 12})
    ax.set_position([0.1, -0.45, 0.8, 2])
    fig.suptitle('GLOW Model Output (2D, geocentric) %s %s'%(day, time_of_day))
    # ax.set_rscale('ofst_r_scale')
    # ax.set_rscale('symlog')
    # ax.set_rorigin(-1)
    plt.savefig('test_geo_%s_%s.pdf'%(wl, file_suffix))
    plt.show()
# %%
def plot_geo_local(bds: xr.Dataset, wl:str, file_suffix: str, *, vmin: float = None, vmax: float = None, decimals: int = 0, num_levels: int = 1000)->None:
    dtime = parse(bds.time).astimezone(get_localzone())
    day = dtime.strftime('%Y-%m-%d')
    time_of_day = dtime.strftime('%H:%M hrs')
    fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(3, 3.2))
    tn = (bds.ver.loc[dict(wavelength=wl)].values).copy()
    r, t = np.meshgrid((bds.alt_km.values + EARTH_RADIUS), bds.angle.values)
    tt, rr = glow2d_polar.get_local_coords(t, r)
    tt = np.pi / 2 - tt
    vidx = np.where(t < 0)
    tn[vidx] = 0
    ticks = None
    levels = num_levels
    if vmin is not None and vmax is not None:
        levels = np.linspace(np.log10(vmin), np.log10(vmax), num_levels, endpoint=True).tolist()
        ticks = np.linspace(np.log10(vmin), np.log10(vmax), 10, endpoint=True)
        ticks = np.unique(np.round(ticks, decimals=decimals))
    im = ax.contourf(tt, rr, np.log10(tn), 100, cmap='gist_ncar_r', levels=levels)
    cbar = fig.colorbar(im, shrink=0.6, ticks=ticks)
    if ticks is not None: cbar.ax.set_yticklabels([r'$10^{%d}$'%(tval) for tval in ticks])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r'\begin{center}%s \AA{} VER ($%s$)\end{center}'%(wl, iono.ver.attrs['units']), fontsize=10)
    ax.set_thetamax(90)
    ax.text(np.radians(-20), ax.get_rmax()/2, 'Distance from observation location (km)',
            rotation=0, ha='center', va='center')
    ax.text(np.radians(90), ax.get_rmax()*1.02, '(Zenith)',
            rotation=0, ha='center', va='center', fontdict={'size': 8})
    fig.suptitle('GLOW Model Output (2D, local polar)\n%s %s'%(day, time_of_day))
    ax.fill_between(np.deg2rad([12, 69]), 0, 10000, alpha=0.3, color='b')
    ax.plot(np.deg2rad([12, 12]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.plot(np.deg2rad([69, 69]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.text(np.deg2rad(37), 1600, r'HiT\&MIS View Cone', fontsize=10, color='w', rotation=360-50)
    ax.tick_params(labelsize=10)
    # earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
    # ax.add_artist(earth)
    # ax.set_thetamax(ang.max()*180/np.pi)
    ax.set_ylim(rr.min(), rr.max())
    # ax.set_rscale('symlog')
    ax.set_rorigin(-rr.min())
    plt.savefig('test_loc_%s_%s.pdf'%(wl, file_suffix))
    plt.show()
# %%
from scipy.signal import savgol_filter
def plot_local(iono: xr.Dataset, wl: str, file_suffix: str, *, vmin: float = None, vmax: float = None, decimals: int = 0, num_levels: int = 1000)->None:
    dtime = parse(iono.time).astimezone(get_localzone())
    day = dtime.strftime('%Y-%m-%d')
    time_of_day = dtime.strftime('%H:%M hrs')
    fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(3, 3.2))
    tn = (iono.ver.loc[dict(wavelength=wl)].values).copy()
    rr, tt = np.meshgrid((iono.r.values), iono.za.values)
    tt = np.pi / 2 - tt
    vidx = np.where(tt < 0)
    tn[vidx] = 0
    ticks = None
    levels = num_levels
    if vmin is not None and vmax is not None:
        levels = np.linspace(np.log10(vmin), np.log10(vmax), num_levels, endpoint=True).tolist()
        ticks = np.linspace(np.log10(vmin), np.log10(vmax), 10, endpoint=True)
        ticks = np.unique(np.round(ticks, decimals=decimals))
    im = ax.contourf(tt, rr, np.log10(tn), cmap='gist_ncar_r', levels=levels)
    cbar = fig.colorbar(im, shrink=0.6, ticks=ticks)
    if ticks is not None: cbar.ax.set_yticklabels([r'$10^{%d}$'%(tval) for tval in ticks])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r'\begin{center}%s \AA{} VER ($%s$)\end{center}'%(wl, iono.ver.attrs['units']), fontsize=10)
    ax.set_thetamax(90)
    ax.text(np.radians(-20), ax.get_rmax()/2, 'Distance from observation location (km)',
            rotation=0, ha='center', va='center')
    ax.text(np.radians(90), ax.get_rmax()*1.02, '(Zenith)',
            rotation=0, ha='center', va='center', fontdict={'size': 8})
    fig.suptitle('GLOW Model Output (2D, local polar)\n%s %s'%(day, time_of_day))
    ax.fill_between(np.deg2rad([12, 69]), 0, 10000, alpha=0.3, color='b')
    ax.plot(np.deg2rad([12, 12]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.plot(np.deg2rad([69, 69]), [0, 10000], lw=0.5, color='k', ls='--')
    ax.text(np.deg2rad(37), 1600, r'HiT\&MIS View Cone', fontsize=10, color='w', rotation=360-50)
    ax.tick_params(labelsize=10)
    # earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
    # ax.add_artist(earth)
    # ax.set_thetamax(ang.max()*180/np.pi)
    ax.set_ylim(rr.min(), rr.max())
    # ax.set_rscale('symlog')
    ax.set_rorigin(-rr.min())
    plt.savefig('test_loc_%s_uniform_%s.pdf'%(wl, file_suffix))
    plt.show()
# %%
for file_suffix in bdss:
    bds = bdss[file_suffix]
    iono = ionos[file_suffix]
    for wl in ('5577', '6300'):
        bds_minmax = get_all_minmax(bdss, 'ver', {'wavelength': wl}, True)
        iono_minmax = get_all_minmax(ionos, 'ver', {'wavelength': wl}, True)
        plot_geo(bds, wl, file_suffix, vmin=bds_minmax[0], vmax=bds_minmax[1])
        plot_geo_local(bds, wl, file_suffix, vmin=bds_minmax[0], vmax=bds_minmax[1])
        plot_local(iono, wl, file_suffix, vmin=1e-3, vmax=iono_minmax[1])
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
ticks = None
levels = int(1e3)
vmin, vmax = gd2.min(), gd2.max()
if vmin is not None and vmax is not None:
    levels = np.linspace(np.log10(vmin), np.log10(vmax), levels, endpoint=True).tolist()
    ticks = np.arange(np.round(np.log10(vmin) + 0.1, decimals=1), np.round(np.log10(vmax), decimals=1), 0.5)
im = ax.contourf(t, r, np.log10(gd2), levels=levels, cmap='gist_ncar_r')
cbar = fig.colorbar(im, shrink=0.6, ticks=ticks)
cbar.set_label('Area Scale', fontsize=10)
cbar.ax.tick_params(labelsize=8)
if ticks is not None: cbar.ax.set_yticklabels([r'$10^{%.1f}$'%(tval) for tval in ticks])
ax.set_thetamax(90)
ax.text(np.radians(-12), ax.get_rmax()/2, 'Distance from observation location (km)',
        rotation=0, ha='center', va='center')
fig.suptitle('Area element scaling from geocentric to local polar coordinates')
ax.fill_between(np.deg2rad([12, 69]), 0, 10000, alpha=0.3, color='b')
ax.plot(np.deg2rad([12, 12]), [0, 10000], lw=0.5, color='k', ls='--')
ax.plot(np.deg2rad([69, 69]), [0, 10000], lw=0.5, color='k', ls='--')
ax.text(np.deg2rad(37), 1600, r'HiT\&MIS View Cone', fontsize=10, color='w', rotation=360-45)
ax.tick_params(labelsize=10)
ax.text(np.radians(90), r.max()*1.02, '(Zenith)',
            rotation=0, ha='center', va='center', fontdict={'size': 8})
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
# ax.set_thetamax(ang.max()*180/np.pi)
ax.set_ylim(r.min(), r.max())
# ax.set_rscale('symlog')
ax.set_rorigin(-60)
plt.savefig('test_loc_geo_distrib.pdf')
plt.show()
# %%
ofst = 1000
scale = 1000
fig = plt.figure(figsize=(3.2, 2.4), dpi=300, constrained_layout=True)
gspec = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gspec[0, 0], projection='polar')
alt = np.linspace(60, 550, 5)
ang = np.linspace(np.deg2rad(2.5), np.deg2rad(27.5), 5) # np.arccos(EARTH_RADIUS/(EARTH_RADIUS + 1000)), 5)
r = (alt + ofst) / scale
# , extent=[0, 0, 7400 / EARTH_RADIUS, ang.max()])
earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
ax.add_artist(earth)
ax.set_thetamax(np.rad2deg(np.arccos(EARTH_RADIUS/(EARTH_RADIUS + 1000))))

cmap = matplotlib.cm.get_cmap('rainbow')

ttext = ('A', 'B', 'C', 'D', 'E')
rtext = ('1', '2', '3', '4', '5')

for tidx, t in enumerate(ang):
    for ridx, dist in enumerate(r):
        col = cmap(1 - ((alt[ridx] - alt.min()) / alt.max()))
        p, _ = glow2d_polar.get_local_coords(t, alt[ridx] + EARTH_RADIUS)
        p = np.pi/2 - p
        ax.scatter(t, dist, s=80, marker='o', facecolors=col if p > 0 else 'w', edgecolors=col, clip_on=False)
        ax.annotate(ttext[tidx] + rtext[ridx], xy=(t - np.deg2rad(0.25), dist), color='w' if ridx ==4 and p > 0 else 'k', weight='heavy', horizontalalignment='center', verticalalignment='center', fontsize='6')

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
ax.set_axisbelow(True)
ax.tick_params(labelsize=10)

# label_position=ax.get_rlabel_position()
ax.text(np.radians(-20), ax.get_rmax()/2, 'Distance from Earth center (km)',
        rotation=0, ha='center', va='center')
ax.set_position([0.1, -0.45, 0.8, 2])
fig.suptitle('Distribution of points in geocentric coordinates')
# fig.suptitle('GLOW Model Output (2D, geocentric) %s %s'%(day, time_of_day))
# ax.set_rscale('ofst_r_scale')
# ax.set_rscale('symlog')
# ax.set_rorigin(-1)
plt.savefig('pt_distrib_geo.pdf')
plt.show()
# %%
from matplotlib import ticker
fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection='polar'), figsize=(3.2, 3.6))

r, t = np.meshgrid(alt + EARTH_RADIUS, ang)
t, r = glow2d_polar.get_local_coords(t, r)
ax.set_ylim(60, r.max())
ax.text(np.radians(90), r.max()*1.02, '(Zenith)',
            rotation=0, ha='center', va='center', fontdict={'size': 8})
ax.text(np.radians(-16), ax.get_rmax()/2, 'Distance from observation location (km)',
        rotation=0, ha='center', va='center')
ax.fill_between(np.deg2rad([12, 69]), 0, 10000, alpha=0.13, color='b')
ax.plot(np.deg2rad([12, 12]), [0, 10000], lw=0.5, color='k', ls='--', alpha=0.5)
ax.plot(np.deg2rad([69, 69]), [0, 10000], lw=0.5, color='k', ls='--', alpha=0.5)

for tidx, t in enumerate(ang):
    for ridx, dist in enumerate(alt):
        col = cmap(1 - ((dist - alt.min()) / alt.max()))
        p, r = glow2d_polar.get_local_coords(t, dist + EARTH_RADIUS)
        p = np.pi/2 - p
        ax.scatter(p, r, s=80, marker='o', facecolors=col if p > 0 else 'w', edgecolors=col, clip_on=True)
        ax.annotate(ttext[tidx] + rtext[ridx], xy=(p, r), color='black' if ridx < 4 else 'w', weight='heavy', horizontalalignment='center', verticalalignment='center', fontsize='6')

# np.meshgrid((alt + ofst) / ofst, ang)
ax.text(np.deg2rad(37), 1600, r'HiT\&MIS View Cone', fontsize=10, color='k', rotation=360-45)
ax.tick_params(labelsize=10)
# earth = pl.Circle((0, 0), 1, transform=ax.transData._b, color='k', alpha=0.4)
# ax.add_artist(earth)
ax.set_thetamax(90)
# ax.set_rscale('symlog')
ax.set_rorigin(-60)
ax.set_axisbelow(True)
ax.tick_params(labelsize=10)
fig.suptitle('Distribution of points in local polar coordinates')

plt.savefig('pt_distrib_local.pdf')
plt.show()
# %% Brightness comparison
def plot_brightness(num_pts: int)->None:
    wls = ('5577', '6300')
    za_min = np.arange(0, 90, 2.5, dtype=float)
    za_max = za_min + 2.5
    za = za_min + 1.25
    za_min = np.deg2rad(za_min)
    za_max = np.deg2rad(za_max)
    tdict = {
        'dawn': datetime(2022, 3, 22, 6, 0).astimezone(pytz.utc),
        'noon': datetime(2022, 3, 22, 12, 0).astimezone(pytz.utc),
        'dusk': datetime(2022, 3, 22, 18, 0).astimezone(pytz.utc),
        'midnight': datetime(2022, 3, 22, 23, 59).astimezone(pytz.utc)
    }
    day: str = ''
    time_of_day: dict(str, str) = {}
    ionos: dict(str, xr.Dataset) = {}
    photonrate: dict(str, dict(str, np.ndarray)) = {}
    photonrate_a: list[np.ndarray] = []
    for key, time in tdict.items():
        iono = polar_model(time, 42.64981361744372, -71.31681056737486, 40, 0, n_pts=num_pts)
        dtime = parse(iono.time).astimezone(get_localzone())
        day = dtime.strftime('%Y-%m-%d')
        time_of_day[key] = dtime.strftime('%H:%M hrs')
        ionos[key] = iono
    for key in tdict.keys():
        photonrate[key] = {}
        for wl in wls:
            em = glow2d_polar.get_emission(ionos[key], feature=wl, za_min=za_min, za_max=za_max)
            photonrate[key][wl] = em
            photonrate_a.append(em.copy())
    vmin, vmax = get_all_minmax_list(photonrate_a) # xmin, xmax
    # vmin = 10**(np.round(np.log10(vmin)) - 1)
    vmax = 10**(np.round(np.log10(vmax)) + 1)
    print(day, time_of_day)
    fig, ax = plt.subplots(dpi=300, figsize=(6.4, 4.8))
    colors = {'dawn': 'k', 'noon': 'r', 'dusk': 'b', 'midnight': 'g'}
    lstyles = {'5577': '-.', '6300': '-'}
    ax.set_title(r'GLOW 2D Photon Emission Rates on %s ($n = %s$)'%(day, num_pts))
    ax.set_xscale('log')
    for key in tdict.keys(): # wish python 3.9 had switch-case
        for wl in wls:
            ax.plot(photonrate[key][wl], za[::-1], ls=lstyles[wl], color=colors[key], lw=0.75)
    # ax.plot(1e4/np.cos(np.deg2rad(90 - za)), za, ls = '--', color='orange', lw=0.75)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(za.min(), za.max())
    ax.set_ylabel('Elevation Angle (deg)')
    ax.set_xlabel(r'Photon Emission Rate ($cm^{-2} rad^{-1} s^{-1}$)')
    ax.text(1e9, 80, 'Dawn', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.text(1e9, 75, 'Noon', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.text(1e9, 70, 'Dusk', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.text(1e9, 65, 'Midnight', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.plot([1.1e9, 2966479394.84], [80, 80], ls = '-', color = colors['dawn'], lw = 0.75)
    ax.plot([2966479394.84, 8e9], [80, 80], ls = '-.', color = colors['dawn'], lw = 0.75)
    ax.plot([1.1e9, 2966479394.84], [75, 75], ls = '-', color = colors['noon'], lw = 0.75)
    ax.plot([2966479394.84, 8e9], [75, 75], ls = '-.', color = colors['noon'], lw = 0.75)
    ax.plot([1.1e9, 2966479394.84], [70, 70], ls = '-', color = colors['dusk'], lw = 0.75)
    ax.plot([2966479394.84, 8e9], [70, 70], ls = '-.', color = colors['dusk'], lw = 0.75)
    ax.plot([1.1e9, 2966479394.84], [65, 65], ls = '-', color = colors['midnight'], lw = 0.75)
    ax.plot([2966479394.84, 8e9], [65, 65], ls = '-.', color = colors['midnight'], lw = 0.75)

    # ax.text(1e9, 40, r'$\csc{\theta}$ \\ (midnight)', horizontalalignment='right', verticalalignment='center')
    # ax.plot([1.1e9, 2966479394.84], [40]*2, ls='--', color='orange', lw=0.75)

    ax.text(1e9, 50, r'5577 \AA{}', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.plot([1.1e9, 2966479394.84], [50]*2, ls = '-.', color = colors['dawn'], lw = 0.75)
    ax.text(1e9, 55, r'6300 \AA{}', fontdict={'size': 10}, horizontalalignment='right', verticalalignment='center')
    ax.plot([1.1e9, 2966479394.84], [55]*2, ls = '-', color = colors['dawn'], lw = 0.75)
    plt.savefig(f'test_prate_{num_pts}.pdf')
    plt.show()

plot_brightness(20)
plot_brightness(50)
plot_brightness(100)
plot_brightness(200)
# %%
