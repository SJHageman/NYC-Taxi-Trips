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
df['bearing'] = bearing(df.pickup_latitude.values, df.pickup_longitude.values, \
  df.dropoff_latitude.values, df.dropoff_longitude.values)
df['distance'] = haversine(df.pickup_latitude.values, df.pickup_longitude.values, \
  df.dropoff_latitude.values, df.dropoff_longitude.values)
df['x'] = df.distance.values*np.cos(np.pi*df.bearing.values/180.0)
df['y'] = df.distance.values*np.sin(np.pi*df.bearing.values/180.0)
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
fig, ax = plt.subplots()
ax.hist2d(df.x.values,df.y.values,bins=3000,cmap='inferno', norm=colors.LogNorm())
ax.set_ylim(-60,60)
ax.set_xlim(-40,40)
#%%
loc_df = pd.DataFrame()
loc_df['longitude'] = df.pickup_longitude
loc_df['latitude'] = df.pickup_latitude

kmeans = KMeans(n_clusters=16, random_state=2, n_init = 10).fit(loc_df)
loc_df['label'] = kmeans.labels_
#%%
loc_df = loc_df.sample(200000)
fig, ax = plt.subplots()
for label in loc_df.label.unique():
    ax.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.3, markersize = 0.3)

ax.set_title('Clusters of New York')

#%%
# Histogramming
r = df.distance.values
theta = np.radians(df.bearing.values)
nr = 2000
ntheta = 360
r_edges = np.linspace(0, 20, nr + 1)
theta_edges = np.linspace(0, 2*np.pi, ntheta + 1)
H, _, _ = np.histogram2d(r, theta, [r_edges, theta_edges])
#%%
# Plot
ax = plt.subplot(polar=True)
Theta, R = np.meshgrid(theta_edges, r_edges)
pcm = ax.pcolormesh(Theta, R, H, norm=colors.LogNorm(), cmap='magma')
plt.colorbar(pcm,ax=ax)
#%%
df_2 = df.sample(100000)
kmeans = KMeans(n_clusters=3, random_state=2, n_init = 10, n_jobs=6, verbose=5).fit(\
			   df_2[['bearing', 'distance']].values)
df_2['label'] = kmeans.labels_
ax = plt.subplot()
for label in df_2.label.unique():
    ax.plot(df_2.bearing[df_2.label == label].values,df_2.distance[df_2.label == label].values,'.', alpha = 0.3, markersize = 0.3)
#%%
df_2 = df.sample(100000)
clusters = DBSCAN(eps=0.3,n_jobs=2).fit(df_2[['x', 'y']].values)
df_2['label']=clusters.labels_
#%%
df_2 = df.sample(200000)
clusters = HDBSCAN(min_cluster_size=750).fit(df_2[['x', 'y']].values)
df_2['label']=clusters.labels_    

#fig, ax = plt.subplots(1,3)
#fig = plt.figure()
#ax1 = plt.subplot(311, projection='polar')
#ax2 = plt.subplot(312)
#ax3 = plt.subplot(313)
fig, ax = plt.subplots(1,3)
for label in df_2.label.unique()[1:]:
	ax[1].plot(df_2.dropoff_longitude[df_2.label == label],df_2.dropoff_latitude[df_2.label == label],'.', alpha = 0.3, markersize = 0.3)
	ax[1].set_title('Dropoff')
	ax[0].plot(df_2.x[df_2.label == label].values,df_2.y[df_2.label == label].values,'.', alpha = 0.3, markersize = 0.3)
	ax[2].plot(df_2.pickup_longitude[df_2.label == label],df_2.pickup_latitude[df_2.label == label],'.', alpha = 0.3, markersize = 0.3)
	ax[2].set_title('Pickup')
#%%
df_2 = df.sample(10000)
clusters = OPTICS(max_eps=0.6,n_jobs=6).fit(df_2[['distance', 'bearing']].values)
df_2['label']=clusters.labels_
    
fig, ax = plt.subplots(1,3)
for label in df_2.label.unique():
	ax[1].plot(df_2.dropoff_longitude[df_2.label == label],df_2.dropoff_latitude[df_2.label == label],'.', alpha = 0.3, markersize = 0.3)
	ax[0].plot(df_2.bearing[df_2.label == label].values,df_2.distance[df_2.label == label].values,'.', alpha = 0.3, markersize = 0.3)
	ax[2].plot(df_2.pickup_longitude[df_2.label == label],df_2.pickup_latitude[df_2.label == label],'.', alpha = 0.3, markersize = 0.3)
#%%
ax = plt.subplot()
ax.plot(df.bearing.values,df.distance.values,'.', alpha = 0.3, markersize = 0.3)


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
	plt.colorbar(pcm, ax=ax.flat[i])
ax.flat[0].set_title('Wednesday 12PM')
ax.flat[1].set_title('Saturday 1AM')
#%%
#fig, axarr = plt.subplots()
#im = axarr.imshow(transmission_image_array[0], aspect='auto', clim=(0,1), cmap=cmapy)
#axarr.axvline(484, ymin=0, ymax=0.5, color='darkorange')
#axarr.axvline(588, ymin=0, ymax=0.5, color='darkgreen')
#
#def update_img(img_array):
#	im.set_data(img_array)
#	return im
#
#ani = animation.FuncAnimation(fig,update_img,transmission_image_array,interval=30)
#%%
d = dwh.get_group((2,0))
r = d.distance.values
theta = np.radians(d.bearing.values)
nr = 200
ntheta = 360
r_edges = np.linspace(0, 20, nr + 1)
theta_edges = np.linspace(0, 2*np.pi, ntheta + 1)
H, _, _ = np.histogram2d(r, theta, [r_edges, theta_edges])

Theta, R = np.meshgrid(theta_edges, r_edges)
fig,ax = plt.subplots(subplot_kw=dict(projection='polar'))

pcm = ax.pcolormesh(Theta, R, H, norm=colors.LogNorm(), cmap='magma')

def update_img(i):
	d = dwh.get_group((2,i))
	r = d.distance.values
	theta = np.radians(d.bearing.values)
	nr = 200
	ntheta = 360
	r_edges = np.linspace(0, 20, nr + 1)
	theta_edges = np.linspace(0, 2*np.pi, ntheta + 1)
	H, _, _ = np.histogram2d(r, theta, [r_edges, theta_edges])
	
	Theta, R = np.meshgrid(theta_edges, r_edges)
	
	pcm = ax.pcolormesh(Theta, R, H, norm=colors.LogNorm(), cmap='magma')
	return pcm

ani = animation.FuncAnimation(fig,update_img,np.arange(0,24),interval=30)