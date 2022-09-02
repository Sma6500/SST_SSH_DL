#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:42:54 2022

@author: lollier
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         MAIN                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

from torch import load, save 
import os
from Trainer import Trainer
from models.UNet import UNet 
from config import unet_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config
from utils.functions import save_config


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

"""
This function takes hyperparameters from config.py. 
It creates an object from the Model class and then uses it to define an object 
from the Trainer class.
The training is launched by the call to the run() method from the trainer object. 
This call is inside a try: block in order to handle exceptions.
For now, the only exception handled is a KeyboardInterrupt: 
the current network will be saved.
"""

def main(UNet, unet_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config):

    print('Building Model...')
    if not('nb_inputs' in unet_config.keys()):
        unet_config['nb_inputs']=2
    dataloader_config['lenght_target']=train_config['prediction_probabilities'][2] if dataloader_config['lenght_target'] is None else dataloader_config['lenght_target']
    net=UNet(unet_config['num_predictions'], in_channels=dataloader_config['lenght_source']*unet_config['nb_inputs'],
             depth=unet_config['depth'], start_filts=unet_config['start_filts'], merge_mode=unet_config['merge_mode'], activation=unet_config['activation'],
             transfo_field=unet_config['transfo_field'], nb_inputs=unet_config['nb_inputs']) #pas élégant, c'est juste pour pouvoir récupérer l'info dans model pour forward
    
    if unet_config['weight_load'] is not(None):
        net.load_state_dict(load(unet_config['weight_load']))
        print("\n weights loaded \n")
        
    trainer = Trainer(net, train_config, dataloader_config, criterion_config, optimizer_config, scheduler_config)
    
    try:
        trainer.run()
    except KeyboardInterrupt:

        net='Unet'
        optimizer='adam'
        basename = net + '_' + optimizer + '.pt'
        filename = 'interrupted_'+ train_config['name'] + basename
        path = os.path.join(train_config['checkpoints_path'],filename)
        save(trainer.model.net.state_dict(), path)
        print()
        print(80*'_')
        print('Training Interrupted')
        print('Current State saved.')
        
    print('Saving config .....')
    dataloader_config.pop('transform', None)
    save_config([unet_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config],
                train_config['checkpoints_path'],
                train_config['name']+'_config')
    path = os.path.join(train_config['checkpoints_path'],'training_finished_'+train_config['name']+'_.pt')
    save(trainer.model.net.state_dict(),path)
        

    

if __name__ == '__main__':

        
    # from torchvision import transforms
    # from utils.functions import crop128

    # if 'nb_training' in train_config.keys():
    #     nb_training=train_config['nb_training']
    # else :
    #     nb_training=1
    
    # for i in range(nb_training):
    #     print(i)
    main(UNet, unet_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config)
        # train_config['name']+=str(i)
        # t=transforms.Compose([transforms.ToTensor(), transforms.Lambda(crop128)])
        # dataloader_config['transform']=t

    
    # #transfo field 4 inputs prediction at time 3

    # train_config['name']='overfit_next_time_step_6_2_'
    # t=transforms.Compose([transforms.ToTensor(), transforms.Lambda(crop128)])
    # dataloader_config['transform']=t
    # unet_config['weight_load']="/usr/home/lollier/datatmp/weight_save/Unet/Unet_simple/overfit_next_time_step_6_1_best_valid_loss.pt"
    # optimizer_config['learning_rate']=0.01
    # criterion_config['Grad_MSE']=0
    # main(UNet, unet_config, dataloader_config, train_config, criterion_config, optimizer_config, scheduler_config)


