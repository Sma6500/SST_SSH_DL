#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:26:13 2022

@author: lollier

Class Trainer that instanciate the classes dataloaders and model in order to achieve training and validation step
We might need to change the functions training_step and validation_step according to the network trained
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         TRAINER                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
from tqdm import tqdm

import torch

from Dataloaders import get_dataloaders
from model import model
from utils.early_stopping import EarlyStopping

import numpy as np
import os

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
class Trainer():
     
    def __init__(self, net, train_config, dataloader_config, criterion_config, optimizer_config, scheduler_config):
        
        self.model = model(net, criterion_config, optimizer_config, scheduler_config) #from class Model
        self.config = train_config #dict, keys are : nb_epochs, checkpoints_path, verbose
        self.trainloader, self.validloader = self.__init_dataloaders__(dataloader_config)
        self.early_stopping = self.__init_early_stopping__()
        self.list_prediction_range=self.update_range_prediction(init=True) #list with the predictions loop probabilities
        self.scheduler_config=scheduler_config
        self.state = {'train_loss': 0,  
                      'valid_loss': 0, 
                      'best_valid_loss': np.Inf,
                      'epoch': 0}
        


    def update_range_prediction(self, init=False, verbose=False):
        """
        Parameters
        ----------
        init : BOOL, optional
            TRUE for initialization. The default is False.
        verbose : BOOL, optional

        Returns
        -------
        LIST
            return the list of probabilities for forecast prediction.
            update the list each epoch according to the validation loss improvement
        """
        
        def create_list_pred(prediction_proba):
            
            list_prediction_range=[]
            
            for i,proba in enumerate(prediction_proba):
                list_prediction_range+=[i]*int(100*proba)
            return list_prediction_range

        if type(self.config['prediction_probabilities']) == list:
            return create_list_pred(self.config['prediction_probabilities'])

        elif init :
            self.alpha, self.alpha_factor, self.range_prediction=self.config['prediction_probabilities']
            return create_list_pred([np.exp(-self.alpha*j) for j in range(self.range_prediction)])
        else :
            if self.state['best_valid_loss']!=np.Inf and self.state['best_valid_loss']/self.state['valid_loss']<1:
                self.alpha*=self.alpha_factor 
                
            list_proba=[np.exp(-self.alpha*j) for j in range(self.range_prediction)]
            if verbose : 
                print(f"next prediction range and probabilities : {list_proba}")
            return create_list_pred(list_proba)
    
    
    #not functionnal yet
    def __str__(self): 
        title = 'Training settings  :' + '\n' + '\n'
        net         = 'Net.......................:  Unet \n' #+ self.model.net + '\n' 
        optimizer   = 'Optimizer.................:  ' + self.model.optimizer + '\n'
        scheduler   = 'Learning Rate Scheduler...:  ' + self.model.scheduler + '\n' #if None we print None
        nb_epochs   = 'Number of epochs..........:  ' + str(self.config['nb_epochs']) + '\n'
        summary = title+ net + optimizer + scheduler + nb_epochs
        return (80*'_' + '\n' + summary + 80*'_')
        
    
    def __init_dataloaders__(self, dataloader_config):
        return get_dataloaders(dataloader_config)

    def __init_early_stopping__(self):
        early_stopping = EarlyStopping(patience=self.config['patience_early_stopping'], 
                                       delta = self.config['delta_early_stopping'], 
                                       verbose=self.config['verbose'])
        early_stopping.set_checkpoints(self.config)
        return early_stopping


    def training_step(self):
        """
        Action
        -------
        Make a training step for an epoch, 
        input sla and sst in the network
        criterion compares sla and sst
        
        Returns
        -------
        train_loss : torch tensor
        """
        
        self.model.net.train()
        
        if self.model.details:
            self.train_loss_details={i:0 for i in self.model.losses_names}
        train_loss = 0

        for item in tqdm(self.trainloader):
            prediction_range=np.random.choice(self.list_prediction_range)
            
            if self.model.details :
                t_l, loss_details = self.model.forward(item, prediction_range)
            else :
                t_l = self.model.forward(item, prediction_range)
            
            train_loss+=t_l
            
            if self.model.details:
                for i in self.train_loss_details.keys():
                    self.train_loss_details[i]+=loss_details[i]
                    
        if self.model.details:
            for i in self.train_loss_details.keys():
                self.train_loss_details[i]=self.train_loss_details[i]/len(self.trainloader)
                
        return train_loss/len(self.trainloader)
    
    
    def validation_step(self):
        """
        Action
        -------
        Make a validation step for an epoch, 
        input sla and sst in the network
        criterion compares sla and sst/ only sla ?
        Returns
        -------
        valid_loss : torch tensor
        """
        
        self.model.net.eval()
        if self.model.details:
            self.valid_loss_details={i:0 for i in self.model.losses_names}
        valid_loss = 0
        with torch.no_grad():
            for item in tqdm(self.validloader):
                prediction_range=np.random.choice(self.list_prediction_range)

                if self.model.details :
                    v_l, loss_details = self.model.forward(item, prediction_range, train=False)
                else :
                    v_l = self.model.forward(item, prediction_range, train=False)
                
                valid_loss+=v_l                         
     
                if self.model.details:
                    for i in self.valid_loss_details.keys():
                        self.valid_loss_details[i]+=loss_details[i]
                    
        if self.model.details:
            for i in self.valid_loss_details.keys():
                self.valid_loss_details[i]=self.valid_loss_details[i]/len(self.validloader)

        return valid_loss/len(self.validloader)
    
    

    def verbose(self):
        print()
        print('Train Loss................: {}'.format(self.state['train_loss']))
        print('Validation Loss.................: {}'.format(self.state['valid_loss']))
        print()
        # lr can't be read easily because ReduceOnPlateau is used
        print('Current Learning Rate.....: {}'.format(self.model.optimizer.param_groups[0]['lr']))
        print('Best Validation Loss........: {}'.format(self.state['best_valid_loss']))
        
    def save_out(self, epoch):
        """
        Input : 
            epoch, int
            current epoch
            
        Action
        -------
        save loss data (with details if required) and learning rate for each epoch
        """
        with open(os.path.join(self.config['checkpoints_path'],self.config['name']+".txt"), "a+") as file :
            file.write('\n\n'+80*'_')
            file.write('\nEPOCH %d / %d' % (epoch+1, self.config['nb_epochs']))
            file.write('\nTrain Loss................: {}'.format(self.state['train_loss']))
            file.write('\nValidation Loss.................: {}'.format(self.state['valid_loss']))
            file.write('\nNext Learning Rate.....: {}'.format(self.model.optimizer.param_groups[0]['lr']))
            
            if self.model.details:
                file.write('\n Details : ')
                for name in self.train_loss_details.keys():
                    file.write('\n Train  '+name+'................: {}'.format(self.train_loss_details[name]))
                    file.write('\n Valid  '+name+'................: {}'.format(self.valid_loss_details[name]))


    def update_state(self):
        """
        Action
        -------
        run training loop for one epoch.
        """
        train_loss = self.training_step()
        valid_loss = self.validation_step()
        if self.scheduler_config['scheduler']=='ROP':
            self.model.scheduler.step(valid_loss)
        else :
            self.model.scheduler.step()
        self.state['train_loss'] = train_loss
        self.state['valid_loss'] = valid_loss
        self.update_range_prediction(verbose=self.config['verbose']) 

        if valid_loss < self.state['best_valid_loss']:
            self.state['best_valid_loss'] = valid_loss
            
            #best valid loss saved with the early stopping
            # filename = 'best_valid_loss'+self.config['name']+'_.pt'
            # path = os.path.join(self.config['checkpoints_path'],filename)
            # torch.save(self.model.net.state_dict(), path)
            # print()
            # print(80*'_')
            # print('Current State saved.')

            
    def run(self):
        """
        Action
        -------
        run training loop 
        if optim True will return std and mean of valid loss
        """
        
        for epoch in range(self.config['nb_epochs']):
            print(80*'_')
            print('EPOCH %d / %d' % (epoch+1, self.config['nb_epochs']))
            self.update_state()
            if self.config['verbose']:
                self.verbose()
            self.save_out(epoch)
            if epoch%self.config['checkpoint']==0:
                filename = 'checkpoint_' + str(epoch)+self.config['name']+'_.pt'
                path = os.path.join(self.config['checkpoints_path'],filename)
                torch.save(self.model.net.state_dict(), path)
                print()
                print(80*'_')
                print('Current State saved.')
            self.early_stopping(self.state['valid_loss'], self.model.net)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                print("Validation loss didn't improve since epoch {}".format(epoch-self.early_stopping.patience))
                break
        
        if self.config['optim']:
            return self.state['best_valid_loss']

if __name__=='__main__':
    
    from models.UNet import UNet
    from config import unet_config, dataloader_config, train_config, criterion_config, scheduler_config, optimizer_config
    
    print('Building Model...')
    dataloader_config['lenght_target']=train_config['prediction_probabilities'][2] if dataloader_config['lenght_target'] is None else dataloader_config['lenght_target']
    
    net=UNet(unet_config['num_predictions'], in_channels=dataloader_config['lenght_source']*2,
             depth=unet_config['depth'], merge_mode=unet_config['merge_mode'], activation=unet_config['activation'])
    
    if unet_config['weight_load'] is not(None):
        net.load_state_dict(torch.load(unet_config['weight_load']))
        print("\n weights loaded \n")
    train_config['prediction_probabilities']=[0,0.5,0.5]
    trainer = Trainer(net, train_config, dataloader_config, criterion_config, optimizer_config, scheduler_config)
    
    trainer.training_step()
    
    
