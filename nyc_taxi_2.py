# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:19:45 2019

@author: Tebe
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from sklearn.cluster import KMeans, DBSCAN, OPTICS
#from math import radians, cos, sin, asin, sqrt, degrees, atan2
import matplotlib.animation as animation
from hdbscan import HDBSCAN


if os.environ['COMPUTERNAME']!='HOBBITON': 
	plt.rcParams['figure.dpi'] = 240

plt.style.use('vibrant')
#import matplotlib as mpl
#plt.rcParams['figure.dpi'] = 240
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

SMALL_SIZE = 11
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def set_size(width, fraction=1, aspect=-1):
#    """ Set aesthetic figure dimensions to avoid scaling in latex.
#
#    Parameters
#    ----------
#    width: float
#            Width in pts
#    fraction: float
#            Fraction of the width which you wish the figure to occupy
#
#    Returns
#    -------
#    fig_dim: tuple
#            Dimensions of figure in inches
#	"""
	# Width of figure
	fig_width_pt = width * fraction

    # Convert from pt to inches
	inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
	golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
	fig_width_in = fig_width_pt * inches_per_pt
	# Figure height in inches
	if aspect==-1:
		fig_height_in = fig_width_in * golden_ratio
	else:
		fig_height_in = fig_width_in * aspect
	fig_dim = (fig_width_in, fig_height_in)

	return fig_dim
width =  433.62001
#%%
fname_train = os.path.join(os.getcwd(),'train.csv')
fname_test = os.path.join(os.getcwd(),'test.csv')

df = pd.read_csv(fname_train)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df['day_of_week'] = df.pickup_datetime.dt.dayofweek
df['hour'] = df.pickup_datetime.dt.hour
df.drop(columns=['id','vendor_id'])
df = df[df.passenger_count != 0]

def  bearing(lat1, long1, lat2, long2):
	b = np.arctan2(np.sin(long2-long1)*np.cos(lat2), np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(long2-long1))
	b = np.degrees(b)
	b = (b + 360) % 360
	return b
def haversine(lat1, lon1, lat2, lon2):

      R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km

      dLat = np.radians(lat2 - lat1)
      dLon = np.radians(lon2 - lon1)
      lat1 = np.radians(lat1)
      lat2 = np.radians(lat2)

      a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
      c = 2*np.arcsin(np.sqrt(a))

      return R * c
  
def speed(distance, duration):
	'''

	Parameters
	----------
	distance : float
		Trip distance calculated from latitude and longitude using Haversine formula.
	duration : float
		Duration of trip in minutes.

	Returns
	-------
	sp : float
		Average speed of trip.

	'''
	sp = distance/duration
	return sp

df['bearing'] = bearing(df.pickup_latitude.values, df.pickup_longitude.values, \
  df.dropoff_latitude.values, df.dropoff_longitude.values)
df['distance'] = haversine(df.pickup_latitude.values, df.pickup_longitude.values, \
  df.dropoff_latitude.values, df.dropoff_longitude.values)
df['x'] = df.distance.values*np.cos(np.pi*df.bearing.values/180.0)
df['y'] = df.distance.values*np.sin(np.pi*df.bearing.values/180.0)
df['speed'] = speed(df.distance, df.trip_duration)
#%%
#xlim = [-74.03, -73.77]
#ylim = [40.63, 40.85]
xlim = [-74.20, -73.68]
ylim = [40.5, 40.92]
df = df[(df.pickup_longitude> xlim[0]) & (df.pickup_longitude < xlim[1])]
df = df[(df.dropoff_longitude> xlim[0]) & (df.dropoff_longitude < xlim[1])]
df = df[(df.pickup_latitude> ylim[0]) & (df.pickup_latitude < ylim[1])]
df = df[(df.dropoff_latitude> ylim[0]) & (df.dropoff_latitude < ylim[1])]
#%%
longitude = list(df.pickup_longitude) + list(df.dropoff_longitude)
latitude = list(df.pickup_latitude) + list(df.dropoff_latitude)
fig,ax = plt.subplots(1,2, sharex=True, sharey=True)
ax[0].hist2d(df.pickup_longitude,df.pickup_latitude,bins=1000, cmap='inferno', norm=colors.LogNorm())
ax[0].set_title('Pickup')
ax[0].set_xlabel('Longitude [deg]')
ax[0].set_ylabel('Latitude [deg]')
ax[1].hist2d(df.dropoff_longitude,df.dropoff_latitude,bins=1000, cmap='inferno', norm=colors.LogNorm())
ax[1].set_title('Dropoff')
ax[1].set_ylabel('Latitude [deg]')
ax[1].set_xlabel('Longitude [deg]')
#%%
dmin = 50.0
spmin = 0.0125
d_mask = (df.distance < dmin)&(df.speed < spmin)&(0.0 < df.distance)&(0.0 < df.speed)
fig, ax = plt.subplots(1,2)
ax[0].hist2d(np.log10(df[d_mask].distance), \
		  df[d_mask].trip_duration,\
			  bins=1000, cmap='inferno', norm=colors.LogNorm())
ax[1].hist2d(np.log10(df[d_mask].distance), \
		  np.log10(df[d_mask].speed),\
			  bins=1000, cmap='inferno', norm=colors.LogNorm())

#%%
fig, ax = plt.subplots()
ax.hist2d(df.x.values,df.y.values,bins=3000,cmap='inferno', norm=colors.LogNorm())
#%%
# Histogramming
r = df.distance.values
theta = np.radians(df.bearing.values)
nr = 1000
ntheta = 1800
r_edges = np.linspace(0, 20, nr + 1)
theta_edges = np.linspace(0, 2*np.pi, ntheta + 1)
H, _, _ = np.histogram2d(r, theta, [r_edges, theta_edges])

# Plot
ax = plt.subplot(polar=True)
Theta, R = np.meshgrid(theta_edges, r_edges)
pcm = ax.pcolormesh(Theta, R, H, cmap='magma', vmin=1, vmax=30)
pcm.cmap.set_under('white')
#fig.colorbar(pcm,ax=ax)
ax.set_ylim(0,1.5)
fig.tight_layout()
#%%
# Histogramming
r = df[d_mask].trip_duration.values
theta = np.radians(df[d_mask].bearing.values)
rmax = 5000
nr = rmax
ntheta = 360
r_edges = np.linspace(0, rmax, nr + 1)
theta_edges = np.linspace(0, 2*np.pi, ntheta + 1)
H, _, _ = np.histogram2d(r, theta, [r_edges, theta_edges])

# Plot
ax = plt.subplot(polar=True)
Theta, R = np.meshgrid(theta_edges, r_edges)
pcm = ax.pcolormesh(Theta, R, H, cmap='magma', vmin=1, vmax=30)
pcm.cmap.set_under('white')
#fig.colorbar(pcm,ax=ax)
# ax.set_ylim(0,1.5)
fig.tight_layout()
#%%
df_2 = df.sample(300000)
clusterer = HDBSCAN(min_cluster_size=200)
clusters = clusterer.fit(df_2[['x', 'y']].values)
df_2['label']=clusters.labels_    
# clusterer.condensed_tree_.plot()

#fig, ax = plt.subplots(1,3)
#fig = plt.figure()
#ax1 = plt.subplot(311, projection='polar')
#ax2 = plt.subplot(312)
#ax3 = plt.subplot(313)
#%%
xxlim = [np.amin(df.x.values),np.amax(df.x.values)]
yylim = [np.amin(df.y.values),np.amax(df.y.values)]
fig, ax = plt.subplots(1,3)
ind = 1
for label in df_2.label.unique()[ind:ind+1]:
	ax[1].plot(df_2.dropoff_longitude[df_2.label == label],df_2.dropoff_latitude[df_2.label == label],'.', alpha = 0.3, markersize = 0.3)
	ax[1].set_title('Dropoff')
	ax[0].plot(df_2.x[df_2.label == label].values,df_2.y[df_2.label == label].values,'.',label=label, alpha = 0.3, markersize = 0.3)
	ax[2].plot(df_2.pickup_longitude[df_2.label == label],df_2.pickup_latitude[df_2.label == label],'.', alpha = 0.3, markersize = 0.3)
	ax[2].set_title('Pickup')
ax[1].set_ylim(ylim)
ax[1].set_xlim(xlim)
ax[2].set_ylim(ylim)
ax[2].set_xlim(xlim)
ax[0].set_ylim(yylim)
ax[0].set_xlim(xxlim)
ax[0].legend(loc='lower right')
#%%
ax = plt.subplot()
for label in df_2.label.unique():
    ax.plot(df_2.x[df_2.label == label].values,df_2.y[df_2.label == label].values,'.', alpha = 0.3, markersize = 0.3)

#%%
dw = df.groupby('day_of_week')

fig, ax = plt.subplots(2,4,subplot_kw=dict(projection='polar'))
#ax = ax.flatten
for i in np.arange(7):
	d = dw.get_group(i)
	# Histogramming
	r = d.distance.values
	theta = np.radians(d.bearing.values)
	nr = 2000
	ntheta = 360
	r_edges = np.linspace(0, 20, nr + 1)
	theta_edges = np.linspace(0, 2*np.pi, ntheta + 1)
	H, _, _ = np.histogram2d(r, theta, [r_edges, theta_edges])
	
	# Plot
	Theta, R = np.meshgrid(theta_edges, r_edges)
	ax.flat[i].pcolormesh(Theta, R, H, norm=colors.LogNorm())
#%%
dwh = df.groupby(['day_of_week','hour'])
wed_9 = dwh.get_group((2,12))
sat_24 = dwh.get_group((6,1))
fig, ax = plt.subplots(1,2,subplot_kw=dict(projection='polar'))
#ax = ax.flatten
for i,d in enumerate([wed_9, sat_24]):
	# Histogramming
	r = d.distance.values
	theta = np.radians(d.bearing.values)
	nr = 200
	ntheta = 360
	r_edges = np.linspace(0, 20, nr + 1)
	theta_edges = np.linspace(0, 2*np.pi, ntheta + 1)
	H, _, _ = np.histogram2d(r, theta, [r_edges, theta_edges])
	
	# Plot
	Theta, R = np.meshgrid(theta_edges, r_edges)
	pcm = ax.flat[i].pcolormesh(Theta, R, H, norm=colors.LogNorm(), cmap='magma')
#	ax.flat[i].set_title('')
	plt.colorbar(pcm, ax=ax.flat[i], orientation='horizontal')
	ax.flat[i].set_ylim(0,5.0)
ax.flat[0].set_title('Wednesday 12PM')
ax.flat[1].set_title('Saturday 1AM')
plt.tight_layout()
#%%





