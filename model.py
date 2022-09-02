#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:00:59 2022

@author: lollier*

model class to pass to the trainer class
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         MODEL                                         | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
from utils.losses import RMSELoss, Grad_MSE, MSE_SLA_Loss, MSE_SST_Loss
import numpy as np
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
class model(): 
    
    def __init__(self, net, criterion_config, optimizer_config, scheduler_config):
        """
        Parameters
        ----------
        net : nn torch network
            torch network from the model folder.
        criterion_config : dictionnary
            see config.py.
        optimizer_config : dictionnary
            see config.py.
        scheduler_config : dictionnary
            see config.py.

        Returns
        -------
        None.

        """
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net.to(self.device)
        self.pred_sst=True if net.num_classes>1 else False
        self.optimizer= self.__init_optimizer__(optimizer_config)
        self.scheduler=self.__init_scheduler__(scheduler_config)
        self.losses, self.losses_weights, self.losses_names=self.__init_criterion__(criterion_config)
        self.details=criterion_config['details']
        if 'true_sst' in criterion_config.keys():
            self.true_sst=criterion_config['true_sst']
        else :
            self.true_sst=False
        
    def criterion(self, predictions, target):
        
        if self.details:
            loss_dict={}
        
        res=0
        for i, (loss, weight, name) in enumerate(zip(self.losses, self.losses_weights, self.losses_names)):
            
            temp=loss(predictions, target)          
            if self.details:
                loss_dict[name]=temp.cpu().detach().item()
            res+=temp*weight
        
        if self.details:
            return res, loss_dict
            
        return res
    
    def __init_criterion__(self, criterion_config):
        
        losses=[]
        weights=[]
        losses_names=[]
        #MSE
        
        if 'MSE' in criterion_config.keys() :
            #SLA
            losses.append(MSE_SLA_Loss())
            weights.append(criterion_config['MSE'][0])
            losses_names.append('MSE_SLA')
            
            #SST
            if criterion_config['MSE'][1]>0 and self.pred_sst:
                losses.append(MSE_SST_Loss())
                weights.append(criterion_config['MSE'][1])
                losses_names.append('MSE_SST')

        if 'RMSE' in criterion_config.keys() :
            losses.append(RMSELoss())
            weights.append(criterion_config['RMSE'])
            losses_names.append('RMSE')
            
        if 'Grad_MSE' in criterion_config.keys() :
            losses.append(Grad_MSE(penalty='l2'))
            weights.append(criterion_config['Grad_MSE'])
            losses_names.append('Grad_MSE')
            
        return losses, weights, losses_names
            
    def __init_optimizer__(self, optimizer_config):
        
        if optimizer_config['optimizer']=='Adam':
            optimizer=torch.optim.Adam(self.net.parameters(), optimizer_config['learning_rate'])
            
        elif optimizer_config['optimizer']=='SGD':
            optimizer=torch.optim.SGD(self.net.parameters(), optimizer_config['learning_rate'])
            
        else : 
            print("optimizer badly configured")
            
        return optimizer
    
    def __init_scheduler__(self, scheduler_config):
        
        if scheduler_config['scheduler']=='ROP':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                   mode=scheduler_config['mode'],
                                                                  factor=scheduler_config['factor'],
                                                                  patience=scheduler_config['patience'],
                                                                  threshold=scheduler_config['threshold'],
                                                                  verbose=scheduler_config['verbose'])
        elif scheduler_config['scheduler']=='ELR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                     gamma=scheduler_config['factor'],
                                                     verbose=scheduler_config['verbose'])
        else : 
            raise ValueError("scheduler badly configured")
            return None
        
        return scheduler
    
    def forward (self, item, prediction_range, train=True):
        
        if self.net.nb_inputs==1:
            return self.forward_sla(item, prediction_range, train=train)
        else :
            return self.forward_glob(item, prediction_range, train=train)        
        
    def forward_glob(self, item, prediction_range, train=True):
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
        
        if train:
            self.optimizer.zero_grad()
        
        prediction = self.net(torch.cat([item['source_sla'], 
                                               item['source_sst']], 
                                              axis=1).to(self.device))
        mem_pred_sla=torch.unsqueeze(torch.clone(prediction[:,0,:,:]),1)
        if self.pred_sst:
            mem_pred_sst=torch.unsqueeze(torch.clone(prediction[:,1,:,:]),1)
        #randomly choose wich range for our predictions, if 0 range t+1, if 1 range t+2

        for i in range(prediction_range):
            
            #NFM : i is always used as i+1 i could modify i range in the for loop
            if self.pred_sst : 
                if self.true_sst:
                    
                    inputs=torch.cat([item['source_sla'][:,i+1:,:,:].to(self.device),
                                                   mem_pred_sla,
                                                   item['source_sst'][:,i+1:,:,:].to(self.device),
                                                   item['target_sst'][:,:i+1,:,:]], axis=1)
                else :
                    inputs=torch.cat([item['source_sla'][:,i+1:,:,:].to(self.device),
                                                   mem_pred_sla,
                                                   item['source_sst'][:,i+1:,:,:].to(self.device),
                                                   mem_pred_sst], axis=1)
                
            else : 
                inputs=torch.cat([item['source_sla'][:,i+1:,:,:].to(self.device),
                                               mem_pred_sla,
                                               item['source_sst'][:,i+1:,:,:].to(self.device),
                                               item['target_sst'][:,:i+1]], axis=1)
            prediction=self.net(inputs)


            mem_pred_sla=torch.cat([mem_pred_sla,torch.unsqueeze(prediction[:,0,:,:],1)], axis=1)
            
            if self.pred_sst:
                mem_pred_sst=torch.cat([mem_pred_sst,torch.unsqueeze(prediction[:,1,:,:],1)], axis=1)

        #loss = self.model.criterion(prediction, torch.unsqueeze(item['target_sla'][:,prediction_range,:,:].to(self.model.device),1))
        if self.pred_sst:
            target=torch.stack([item['target_sla'][:,prediction_range,:,:], item['target_sst'][:,prediction_range,:,:]],1).to(self.device)
        else : 
            target=torch.unsqueeze(item['target_sla'][:,prediction_range,:,:], axis=1).to(self.device)
        if self.details:
            loss, loss_details = self.criterion(prediction, target)
        else :   
            loss = self.criterion(prediction, target)
            
        if train :
            loss.backward()
            self.optimizer.step()
        
        if self.details :
            return loss.item(), loss_details
        else :
            return loss.item()
        
    def forward_sla(self, item, prediction_range, train=True):
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
        
        if train:
            self.optimizer.zero_grad()
        
        prediction = self.net(item['source_sla'].to(self.device))
        mem_pred_sla=torch.unsqueeze(torch.clone(prediction[:,0,:,:]),1)

        for i in range(prediction_range):
            
            #NFM : i is always used as i+1 i could modify i range in the for loop
            inputs=torch.cat([item['source_sla'][:,i+1:,:,:].to(self.device),
                                           mem_pred_sla], axis=1)

            prediction=self.net(inputs)

            mem_pred_sla=torch.cat([mem_pred_sla,torch.unsqueeze(prediction[:,0,:,:],1)], axis=1)
            
        #loss = self.model.criterion(prediction, torch.unsqueeze(item['target_sla'][:,prediction_range,:,:].to(self.model.device),1))
        target=torch.unsqueeze(item['target_sla'][:,prediction_range,:,:], axis=1).to(self.device)
        if self.details:
            loss, loss_details = self.criterion(prediction, target)
        else :   
            loss = self.criterion(prediction, target)
            
        if train :
            loss.backward()
            self.optimizer.step()
        
        if self.details :
            return loss.item(), loss_details
        else :
            return loss.item()