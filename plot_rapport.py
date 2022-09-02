#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:21:18 2022

@author: lollier
"""

import torch
import numpy as np
import os
from utils.plot import *
from torchvision import transforms
from utils.functions import *
from Dataloaders import time_serie_dataset
from models.UNet import UNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import matplotlib.pyplot as plt
from model import model
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
#plot loss training

path_data='/usr/home/lollier/datatmp/npy_data/'

#%%

train_valid_sla=np.load(os.path.join(path_data,'train/sla_ts_residus.npy'))
test_sla=np.concatenate([np.load(os.path.join(path_data,'test/sla_ts_residus_1993.npy')),np.load(os.path.join(path_data,'test/sla_ts_residus_2019.npy'))])
train_sla=train_valid_sla[:-1315]
valid_sla=train_valid_sla[-1315:]

train_valid_sst=np.load(os.path.join(path_data,'train/sst_ts_residus.npy'))
test_sst=np.concatenate([np.load(os.path.join(path_data,'test/sst_ts_residus_1993.npy')),np.load(os.path.join(path_data,'test/sst_ts_residus_2019.npy'))])
train_sst=train_valid_sst[:-1315]
valid_sst=train_valid_sst[-1315:]

fig, axs=plt.subplots(3,2, figsize=(20,30))
axs[0,0].hist(train_sla.flatten(), bins=50)
axs[0,0].set_title('SSH (m)')
axs[0,1].hist(train_sst.flatten(), bins=50)
axs[0,1].set_title('SST (C°)')
axs[1,0].hist(valid_sla.flatten(), bins=50)
axs[1,1].hist(valid_sst.flatten(), bins=50)
axs[2,0].hist(test_sla.flatten(), bins=50)
axs[2,1].hist(test_sst.flatten(), bins=50)

#%%

sst=np.load(os.path.join(path_data,'sst_ts.npy'))

plt.hist([sst[:,:108,:].flatten(),sst[:,108:,:].flatten()],bins=50)
plt.legend(['latitude [degrees_north]: 26.5-35.5','latitude [degrees_north]: 35.5-45.5'])
plt.title('SST values histogram divided by latitude')
#%%
ssh=np.load(os.path.join(path_data,'sla_ts.npy'))

plt.hist([ssh[:,:108,:].flatten(),ssh[:,108:,:].flatten()],bins=50)
plt.legend(['latitude [degrees_north]: 26.5-35.5','latitude [degrees_north]: 35.5-45.5'])
plt.title('SSH values histogram divided by latitude')
plt.show()

ssh_res=np.load(os.path.join(path_data,'sla_ts_residus.npy'))

plt.hist([ssh_res[:,:108,:].flatten(),ssh_res[:,108:,:].flatten()],bins=50)
plt.legend(['latitude [degrees_north]: 26.5-35.5','latitude [degrees_north]: 35.5-45.5'])
plt.title('SSH residuals values histogram divided by latitude')

#%%
from statsmodels.tsa.seasonal import seasonal_decompose

ssh=np.load(os.path.join(path_data,'sla_ts.npy'))
decomposition=seasonal_decompose(ssh[:,64,64],period=365)

year=np.linspace(1993,2019,9497)
trend=decomposition.trend[~np.isnan(decomposition.trend)]
season=decomposition.seasonal[~np.isnan(decomposition.seasonal)]
season=season[182:-182]
res=decomposition.resid[~np.isnan(decomposition.resid)]

fig, axs=plt.subplots(3,1, figsize=(15,20), sharex=True)
axs[0].plot(year,trend)
axs[0].set_title("Tendance, saisonnalité et résidus d'un exemple de SSH \n           Tendance")
axs[1].plot(year, season)
axs[1].set_title('Saisonnalité')
axs[2].plot(year, res)
axs[2].set_title('Résidus')
axs[2].set_xlabel('year')


#%%

array=ssh[445,:,:]
array, axes_label=check_shape(array)
fig, ax=plt.subplots()
img = ax.imshow(np.flipud(array), cmap='seismic', extent=axes_label)
ax.set(xlabel='Longitude', ylabel='Latitude')
ax.set_title("Studied SSH area")
rect = patches.Rectangle((-64.25, 35), 10.7, 10.7, linewidth=1, edgecolor='black', facecolor='none')
fig.colorbar(img, ax=ax)
ax.add_patch(rect)
plt.show()