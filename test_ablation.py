#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:34:32 2022

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

from tqdm import tqdm
#plot loss training

path_train='/usr/home/lollier/datatmp/weight_save/Unet/Ablation/'
trains=os.listdir(path_train)

path_data_1993=["/usr/home/lollier/datatmp/npy_data/test/sla_ts_residus_1993.npy","/usr/home/lollier/datatmp/npy_data/test/sst_ts_residus_smooth_1993.npy"]
path_data_2019=["/usr/home/lollier/datatmp/npy_data/test/sla_ts_residus_smooth_2019.npy","/usr/home/lollier/datatmp/npy_data/test/sst_ts_residus_smooth_2019.npy"]

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("data loading\n")

t=transforms.Compose([transforms.ToTensor(), transforms.Lambda(crop128)])
                      
dataloader_config={'dataset_path':["/data/lollier/train/sla_ts_residus.npy","/data/lollier/train/sst_ts_residus.npy"],
                   'lenght_source': 4, #how many timestep in inputs
                   'lenght_target': 3, #how many timestep for prediction (if None monitered with prediction probabilities)
                   'timestep':5, #days between each inputs
                   'transform':t,
                   'valid_ratio':0, 
                   'batch_size': 1,
                   'normalize':True,
                   'small_train': False} #if True take only half train data

datasets=[np.load(i) for i in path_data_2019]

max_sla=np.max(datasets[0])
max_sst=np.max(datasets[1])

min_sla=np.min(datasets[0])
min_sst=np.min(datasets[1])
    
datasets=[minmax_scaler(data) for data in datasets]
    
test_set=time_serie_dataset(datasets, 
                            dataloader_config['lenght_source'],
                            dataloader_config['lenght_target'],
                            dataloader_config['timestep'],
                            dataloader_config['transform'])

sampler=SequentialSampler(test_set)

test_loader=DataLoader(test_set, sampler=sampler)    



################################################################################################################################

def forward (net, item, prediction_range, pred_sst):
    
    if net.nb_inputs==1:
        return forward_sla(net, item, prediction_range)
    else :
        return forward_glob(net, item, prediction_range, pred_sst)        
    
def forward_glob(net, item, prediction_range, pred_sst):
    """
    Parameters
    ----------
    item : dictionnary
        inputs with sla, sst, u, v.
    prediction_range : int
        range of prediction
    train : Bool, optional
        if false don't reset the grad. The default is True.

    Returns
    -------
    TYPE
        loss.
    Actions
    -------
    performs one step through the neural net 
    """
    
    prediction = net(torch.cat([item['source_sla'], 
                                item['source_sst']], axis=1).to(device))
    
    mem_pred_sla=torch.unsqueeze(torch.clone(prediction[:,0,:,:]),1)
    
    if pred_sst:
        mem_pred_sst=torch.unsqueeze(torch.clone(prediction[:,1,:,:]),1)
    #randomly choose wich range for our predictions, if 0 range t+1, if 1 range t+2

    for i in range(prediction_range):
        
        #NFM : i is always used as i+1 i could modify i range in the for loop
        if pred_sst : 
            inputs=torch.cat([item['source_sla'][:,i+1:,:,:].to(device),
                                           mem_pred_sla,
                                           item['source_sst'][:,i+1:,:,:].to(device),
                                           mem_pred_sst], axis=1)
            
        else : 
            inputs=torch.cat([item['source_sla'][:,i+1:,:,:].to(device),
                                           mem_pred_sla,
                                           item['source_sst'][:,i+1:,:,:].to(device),
                                           item['target_sst'][:,:i+1].to(device)], axis=1)
        prediction=net(inputs)

        mem_pred_sla=torch.cat([mem_pred_sla,torch.unsqueeze(prediction[:,0,:,:],1)], axis=1)
        
        if pred_sst:
            mem_pred_sst=torch.cat([mem_pred_sst,torch.unsqueeze(prediction[:,1,:,:],1)], axis=1)

    return torch.squeeze(prediction).cpu().detach().numpy()
    

def forward_sla(net, item, prediction_range):
    """
    Parameters
    ----------
    item : dictionnary
        inputs with sla, sst, u, v.
    prediction_range : int
        range of prediction
    train : Bool, optional
        if false don't reset the grad. The default is True.

    Returns
    -------
    TYPE
        loss.
    Actions
    -------
    performs one step through the neural net 
    """
    
    prediction = net(item['source_sla'].to(device))
    
    mem_pred_sla=torch.unsqueeze(torch.clone(prediction[:,0,:,:]),1)

    for i in range(prediction_range):
        
        #NFM : i is always used as i+1 i could modify i range in the for loop
        inputs=torch.cat([item['source_sla'][:,i+1:,:,:].to(device),
                                       mem_pred_sla], axis=1)

        prediction=net(inputs)

        mem_pred_sla=torch.cat([mem_pred_sla,torch.unsqueeze(prediction[:,0,:,:],1)], axis=1)
        
    return torch.squeeze(prediction).cpu().detach().numpy()

def denormalize(data, cat='sla'):
    """
    Parameters
    ----------
    data : torch tensor (128,128)
        image to unnormalize.

    Raises
    ------
    ValueError
        if shape or type is not correct

    Returns
    -------
    data : torch tensor (128,128)
        image without normalization
    """
    
    data=np.squeeze(data)
    if data.shape!=(128,128):
        print(data.shape)
        raise ValueError("unvalid shape and or unvalid type")
    
    if cat=='sla':
        data=(data*(max_sla-min_sla))+min_sla
    elif cat=='sst':
        data=(data*(max_sst-min_sst))+min_sst
    else :
        raise ValueError("unvalid cat, needs sst or sla")
    return data

def nse(predictions, targets):
    return (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))
################################################################################################################################

# for net_name in trains:
for prediction_range in [1,2,3]:
    print(f"\n______________________________________________\n{prediction_range}\n__________________________________________________________________\n")
    for net_name in ['sla_sst_tf']:#os.listdir(path_train):
        path=os.path.join(path_train, net_name)
        weights=os.listdir(path)
        path_config=os.path.join(path, net_name+'__config.json')
                             
        unet_config, _, train_config, criterion_config, scheduler_config, optimizer_config = load_config(path_config, t)
    
        print('Building Model...')
        if not('nb_inputs' in unet_config.keys()):
            unet_config['nb_inputs']=2
            
        net=UNet(unet_config['num_predictions'], 
                 in_channels=dataloader_config['lenght_source']*unet_config['nb_inputs'],
                 depth=unet_config['depth'],
                 start_filts=unet_config['start_filts'], 
                 merge_mode=unet_config['merge_mode'], 
                 activation=unet_config['activation'], 
                 transfo_field=unet_config['transfo_field'],
                 nb_inputs=unet_config['nb_inputs'])
        net=net.to(device)
        pred_sst=True if net.num_classes>1 else False
        pred_sst=False
        image_true=[]
        for item in test_loader:
            if pred_sst:
                image_true.append(np.stack([denormalize(item['target_sla'][0,prediction_range-1,:,:].numpy()),
                                           denormalize(item['target_sst'][0,prediction_range-1,:,:].numpy(), cat='sst')], axis=0))
            else: 
                image_true.append(denormalize(item['target_sla'][:,prediction_range-1,:,:].numpy()))
                
        image_preds={}
        image_pred=[]
        print(net_name)
        for weight in weights:
            if weight[-18:]=='best_valid_loss.pt':
                
                path_weight_load=os.path.join(path, weight)
                net.load_state_dict(torch.load(path_weight_load, map_location=device))
                image_preds[weight]=[]
    
                with torch.no_grad():
                    for item in tqdm(test_loader):
                        
                        #pred_sst=True if net.num_classes>1 else False
                        pred_sst=False
                        prediction=forward(net, item, prediction_range, pred_sst)
                        #pred_sst=True
                        if pred_sst:
                            image_preds[weight].append(denormalize(prediction[0,:,:]))
                            #image_preds[weight].append(np.stack([denormalize(prediction[0,:,:]),
                                                                 #denormalize(prediction[1,:,:], cat='sst')], axis=0))
                        else:
                            image_preds[weight].append(denormalize(prediction[0,:,:]))
                            
        for i in range(len(test_set)):
            image_pred.append(np.mean(np.array([image_preds[key][i] for key in image_preds.keys()]), axis=0))
        
        nse_coeffs=[]
        rmse_coeffs=[]
        for i in range(len(test_set)):
            nse_coeffs.append(nse( image_pred[i],image_true[i]))
            rmse_coeffs.append(rmse( image_pred[i],image_true[i]))
        plot_pred(image_true[450+(prediction_range*5)],image_true[450], image_pred[450])
        print(f"mean Nash :{np.mean(nse_coeffs)}, mean rmse : {np.mean(rmse_coeffs)}")





































