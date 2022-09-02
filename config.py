#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:13:27 2022

@author: lollier

config file
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         CONFIG                                        | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
from torchvision import transforms
from utils.functions import crop128
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             ONLY THIS FILE SHOULD BE MODIFIED                         | #
# |                           ALL HYPER PARAMETERS SHOULD BE CONFIGURED                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #





######################################### DATALOADER ##########################################


t=transforms.Compose([transforms.ToTensor(), transforms.Lambda(crop128)])#, 
                      #transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC)])
                      
dataloader_config={'dataset_path':["/usr/home/lollier/datatmp/npy_data/train/sla_ts_residus.npy","/usr/home/lollier/datatmp/npy_data/train/sst_ts_residus.npy"],
                   'lenght_source': 4, #how many timestep in inputs
                   'lenght_target': None, #how many timestep for prediction (if None monitered with prediction probabilities)
                   'timestep':5, #days between each inputs
                   'transform':t,
                   'valid_ratio':0.15, 
                   'batch_size': 32,
                   'normalize':True,
                   'small_train': False} #if True take only half train data

########################################### TRAIN #############################################
    
train_config = {
    'nb_epochs' : 500, # arbitrary high
    'checkpoints_path': '/usr/home/lollier/datatmp/weight_save/', 
    'verbose': True,
    'checkpoint':1000, #save the weights every x epochs
    'name':'sla_sst_tf_future_w_loaded',
    'patience_early_stopping':300, #needs to be >>> patience of lr scheduler
    'delta_early_stopping':0.00000001,
    'prediction_probabilities':(0.01,0.9,3), #if tuple : (alpha, alpha factor, range forecast);if list :proba of prediction loop (if [0.5,0.5] training and validation will pred t+2 50% of time)
    'optim':False, #if True, trainer will return mean valid loss and std valid loss for optimization
    'nb_training':1
}

########################################### Model #############################################

unet_config = {
    'num_predictions' : 2,
    'depth':5, 
    'merge_mode': 'concat', 
    'weight_load':"/usr/home/lollier/datatmp/weight_save/Unet/Ablation/sla_sst_tf/sla_sst_tf_01best_valid_loss.pt",
    'activation':'SiLU', #SiLU or ReLU
    'transfo_field':True,
    'start_filts':64,
    'nb_inputs':2,
}
########################################### criterion config #############################################

criterion_config = {
    'MSE' : (1.0,1.0), # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
    'Grad_MSE':0.0,
    'details':True,
    'true_sst':False
}

########################################### criterion config #############################################

optimizer_config = {
    'optimizer' : 'Adam', # eventual features and weights of losses (put None instead of 0 if you don't want the loss to be compute)
    'learning_rate' : 0.01
}

######################################### SCHEDULER ###########################################

scheduler_config = {
    'scheduler': 'ROP', # ReduceOnPlateau: when the loss isn't decreasing for too long, reduce lr only ROP is configure for now, ELR
    'mode': 'min', # we want to detect a decrease and not an increase. 
    'factor': 0.5, # when loss has stagnated for too long, new_lr = factor*lr
    'patience': 20,# how long to wait before updating lr
    'threshold': 0.00000001, 
    'verbose': False
}
