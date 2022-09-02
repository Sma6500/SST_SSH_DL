#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:38:56 2022

@author: lollier
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Functions                                     | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import json
import os 
from torchvision.transforms.functional import crop
import torch

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   UTILS FUNCTIONS TO PROCESS DATA, SAVE AND LOAD RESULT               | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

'''
w  write mode
r  read mode
a  append mode

w+  create file if it doesn't exist and open it in write mode
r+  open for reading and writing. Does not create file.
a+  create file if it doesn't exist and open it in append mode
'''

def save_dict(dictionary, path, filename):
    """
    Parameters
    ----------
    dictionary : type : guess :p
        config dictionary.
    path : str
        save config path.
    filename : str
        name of config saving file.

    Returns
    -------
    None.

    """
    
    file = open(os.path.join(path, filename) +".json", "a+")
    json.dump(dictionary, file)
    file.close()

def save_config(config_list, path, filename):
    """
    Parameters
    ----------
    config_list : list of dictionary
        all config dictionary.
    path : str
        save config path.
    filename : str
        name of config saving file.

    Returns
    -------
    None.

    """
    
    save_dict(json.dumps([config for config in config_list]), path, filename)
    
def load_config(path_config, transformation):
    """
    Parameters
    ----------
    path_config : str
        path config.
    transformation : torch transform
        transformation to apply to tensor in order to go through the network.

    Returns
    -------
    model_config : dict
        DESCRIPTION.
    dataloader_config : dict
        DESCRIPTION.
    train_config : dict
        DESCRIPTION.
    criterion_config : dict
        DESCRIPTION.
    scheduler_config : dict
        DESCRIPTION.
    optimizer_config : dict
        DESCRIPTION.

    """
    
    with open(path_config, "r") as config_file:
        configs=json.loads(json.load(config_file))
        
        model_config=configs[0]
        
        dataloader_config=configs[1]
        dataloader_config['transform']=transformation
        
        train_config=configs[2]
        
        criterion_config=configs[3]
        
        scheduler_config=configs[4]
        
        optimizer_config=configs[5]
    
    return model_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config

def minmax_scaler(data):
    """
    Parameters
    ----------
    data : numpy array
        data to normalize
    Returns
    -------
    data : numpy array
        data normalize following a min max algorithm.

    """
    
    maximum=np.max(data)
    minimum=np.min(data)
    ecart=maximum-minimum
    data=(data-minimum)/ecart

    return data.astype('float32')
    
def crop128(image):
    """
    Parameters
    ----------
    image : tensor 
        sst or sla (216,270) or bigger image.
        
    Returns
    -------
    tensor
        left high corner cropped to (128,128).
    """
    return crop(image, 0, 0, 128, 128)
 
    
if __name__=="__main__":
    path='/usr/home/lollier/datatmp/weight_save/Unet'
    filename='test'
    train_config = {
        'nb_epochs' : 1000, # arbitrary high
        'checkpoints_path': '/usr/home/lollier/datatmp/weight_save/Unet', 
        'verbose': True,
        'checkpoint':50, #save the weights every 50 epochs
        'name':'transfo_field',
        'patience_early_stopping':50, #needs to be > patience of lr scheduler
        'delta_early_stopping':0.00001
    }
    
    save_dict(train_config, path, filename)