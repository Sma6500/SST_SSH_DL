#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:48:36 2022

@author: lollier
"""


import torch

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,prediction,target):
        return torch.sqrt(self.mse(prediction,target))

#needs to create two MSE loss to print them separately
class MSE_SLA_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,prediction,target):
        return self.mse(prediction[:,0,:,:],target[:,0,:,:])

class MSE_SST_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self,prediction,target):
        return self.mse(prediction[:,1,:,:],target[:,1,:,:])
    
class Grad_MSE(torch.nn.Module):
    """
    N-D gradient loss.
    We overwrite this loss for a 2D input 
    """

    def __init__(self, penalty='l2'):
        super().__init__()
        self.penalty = penalty
        self.mse=torch.nn.MSELoss()
        
    def forward(self, prediction, target): 
            
        dy_pred = torch.abs(prediction[:, 0, 1:, :] - prediction[:, 0, :-1, :]) #0 for channel because only sla grad is considered
        dx_pred = torch.abs(prediction[:, 0, :, 1:] - prediction[:, 0, :, :-1])
        
        dy_targ = torch.abs(target[:, 0, 1:, :] - target[:, 0, :-1, :])
        dx_targ = torch.abs(target[:, 0, :, 1:] - target[:, 0, :, :-1])    
        
        if self.penalty == 'l2':
            dy_pred = dy_pred**2
            dx_pred = dx_pred**2
            dy_targ= dy_targ **2
            dx_targ = dx_targ**2
            
        grad_loss=self.mse(torch.cat([dy_pred.flatten(),dx_pred.flatten()]), torch.cat([dy_targ.flatten(),dx_targ.flatten()]))
        
        return grad_loss