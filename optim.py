#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:13:08 2022

@author: lollier
"""




# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         OPTIM                                         | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import numpy as np


from models.Unet_simple import UNet 
from config import unet_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config
from torch import load, save 
import os
from Trainer import Trainer

import matplotlib.pyplot as plt

from ax.service.managed_loop import optimize
from ax.storage.json_store.save import save_experiment
from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour, interact_contour
from ax.utils.notebook.plotting import render

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                 THIS FILES LAUNCH A BAYESIAN OPTIM ON HYPERPARAMETERS                 | #
# |                   OPTIM NEEDS TO BE SET TO TRUE IN THE config.py FILE                 | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


def update_config(new_config, config_list):
    """
    Parameters
    ----------
    new_config : dict
        dict of parameters update.
    config_list : list dict
        all the dict from config.py.

    Actions
    -------
    plot the update of parameters

    """
    for key in new_config.keys():
        for config in config_list:
            if key in config.keys():
                config[key]=new_config[key]
                print(f"{key} has been updated")
                
def train_evaluate(parameterization):
    """
    Parameters
    ----------
    parameterization : dict
        dict of parameters update.

    Actions
    -------
    similar to main.py, run a training on given parameters

    """
    
    print("\n start training ....")
    print(f"parameterization : {parameterization}")
    
    parameterization['prediction_probabilites']=(parameterization['alpha'], 1., 5)
    update_config(parameterization, [unet_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config])
    dataloader_config['lenght_target']=train_config['prediction_probabilities'][2] if dataloader_config['lenght_target'] is None else dataloader_config['lenght_target']
    
    net=UNet(unet_config['num_predictions'], in_channels=dataloader_config['lenght_source']*2,
              depth=unet_config['depth'], start_filts=unet_config['start_filts'],merge_mode=unet_config['merge_mode'], activation=unet_config['activation'],
              transfo_field=unet_config['transfo_field'])

    trainer = Trainer(net, train_config, dataloader_config, criterion_config, optimizer_config, scheduler_config)
        
    mean_valid_loss, std_valid_loss = trainer.run()
    
    print(f"mean_valid_loss :{mean_valid_loss} \nstd_valid_loss : {std_valid_loss}")
    print("\n end training ....")
    
    return {"MSE" :(mean_valid_loss, std_valid_loss)}

if __name__=='__main__':
    
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "learning_rate", "type": "range", "value_type":"float", "bounds": [1e-4, 1.0], "log_scale": True},
            {"name": "patience ", "type": "range", "value_type":"int", "bounds": [5, 20]},
            {"name": "factor", "type": "range", "value_type":"float","bounds": [0.1,0.9]},
            {"name": "alpha", "type": "range", "bounds":[1.,20.]},
            {"name":"batch_size", "type":"choice", "values":[16,32,64]},
            {"name":"depth", "type":"choice", "values":[3,4,5]},
            {"name":"activation","type":"choice","values":["ReLU","SiLU"]}
        ],
        evaluation_function=train_evaluate,
        objective_name='MSE',
        minimize=True,
        total_trials=50,
        arms_per_trial=3,
    )
    print(f"best parameters :{best_parameters}")
    print(f"mean and covariance : {values}")
        
    save_experiment(experiment, "/usr/home/lollier/datatmp/weight_save/Unet/Optim/hyperparameters_optim.json")
    
    plot_contour(model=model, param_x='alpha', param_y='patience', metric_name='MSE')
    plt.savefig("/usr/home/lollier/datatmp/weight_save/Unet/Optim/0.png")
    
    plot_contour(model=model, param_x='learning_rate', param_y='alpha', metric_name='MSE')
    plt.savefig("/usr/home/lollier/datatmp/weight_save/Unet/Optim/1.png")

    plot_contour(model=model, param_x='learning_rate', param_y='factor', metric_name='MSE')
    plt.savefig("/usr/home/lollier/datatmp/weight_save/Unet/Optim/2.png")

    render(plot_contour(model=model, param_x='alpha', param_y='patience', metric_name='MSE'))
    plt.savefig("/usr/home/lollier/datatmp/weight_save/Unet/Optim/00.png")
    
    render(plot_contour(model=model, param_x='learning_rate', param_y='alpha', metric_name='MSE'))
    plt.savefig("/usr/home/lollier/datatmp/weight_save/Unet/Optim/11.png")

    render(plot_contour(model=model, param_x='learning_rate', param_y='factor', metric_name='MSE'))
    plt.savefig("/usr/home/lollier/datatmp/weight_save/Unet/Optim/22.png")

    