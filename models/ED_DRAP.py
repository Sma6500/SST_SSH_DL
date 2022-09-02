#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:08:54 2022

@author: Luther Ollier from paper :
    H. Che, D. Niu, Z. Zang, Y. Cao and X. Chen, "ED-DRAP: Encoder–Decoder Deep Residual Attention Prediction Network for Radar Echoes," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 1004705, doi: 10.1109/LGRS.2022.3141498.

- Unet related modules
- Attentions modules
- ED-DRAP

"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         ED-DRAP                                       | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import numpy as np

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        padding_mode='replicate',
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def batch_norm(in_channels):
    return nn.BatchNorm3d(in_channels)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU/SiLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, activation=nn.ReLU()):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.norm = batch_norm(self.out_channels)
        self.activation=activation

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):

        x = self.activation(self.norm(self.conv1(x)))
        x = self.activation(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool
    
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                        Attention modules                              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

class SEA(nn.Module):
    """
    3-D Sequence attention module SEA : 
    performs 3-D average pooling to contract spatial info and keep temporal, two conv 3D and sigmoid 
    join a weight to each time step in order to capture the most important time related features
    """
    
    def __init__(self, channels, image_size):
        
        
        self.channels=channels
        self.kernel_size=(1,image_size, image_size)
        self.average_pool=nn.AvgPool3d(kernel_size=self.kernel_size) #1,ikmage size ?
        self.conv1=conv1x1(self.channels, 2*self.channels)
        self.conv2=conv1x1(2*self.channels, self.channels)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x):
        
        out=self.average_pool(x)
        out=self.conv1(out)
        out=self.conv2(out)
        out=self.sigmoid(out)
        
        return out+x
              
            
class SPA(nn.Module):
    """
    3-D Spatial attention module SPA : 
    performs 3-D conv to contract time related information and sigmoid to join a 
    weight to each pixel to capture the most important spatial features
    """
    def __init__(self, channels, time_size):
        
        self.channels=channels
        self.kernel=(time_size,1,1)
        self.conv=nn.Conv3d(self.channels, self.channels, self.kernel_size) #10,1,1
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        
        out=self.sigmoid(self.conv(x))
        
        return out+x
        
class SSAB(nn.Module):
    """
    Sequence and Spatial Attention Block
    """
    def __init__(self, channels, input_size):
        
        self.channels=channels
        self.time_size=input_size[0]
        
        if input_size[1]!=input_size[2]:
            raise ValueError(f"image size is not squared anymore {input_size}")
            
        self.image_size=input_size[1]
        
        self.conv1 = conv3x3(self.channels, self.channels)
        self.conv2 = conv3x3(self.channels, self.channels)
        self.sea = SEA(self.channels, self.image_size)
        self.spa = SPA(self.channels, self.time_size)
        
    def forward(self, x):
        
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.sea(out)
        out=self.spa(out)
        
        return out+x
    

class RSSAB(nn.Module):
    """
    performs M-SSAB
    the original paper doesnt mention any number so it's basically a parameter to tune
    """
    def __init__(self, n_ssab, channels, input_size):
        
        self.input_size=input_size #note : si je sépare input_size direct ici je gagne un peu en complexité
        self.n_ssab=n_ssab
        self.channels=channels
        
        self.ssabs=[]
        for i in range(self.n_ssab):
            ssab=SSAB(self.channels, self.input_size)
            self.ssabs.append(ssab)
        
        self.conv=conv3x3(self.channels, self.channels)
        self.ssabs=nn.ModuleList(self.ssabs)
        
    def forward(self, x):
        
        for i, module in enumerate(self.ssabs):
            out=module(x) if i==0 else module(out)
        
        out=self.conv(out)
        
        return out+x

class N_RSSAB(nn.Module):
    """
    performs N RSSAB
    """
    def __init__(self, n_rssab, channels, input_size, n_ssab=4):#ya marqué nulle part dans le papier combien de ssab ils mettent
        
        self.input_size=input_size #note : si je sépare input_size direct ici je gagne un peu en complexité
        self.n_rssab=n_rssab
        self.channels=channels
        self.n_ssab=n_ssab
        
        self.rssabs=[]
        for i in range(self.n_rssab):
            rssab=RSSAB(self.channels, self.input_size, self.n_ssab)
            self.rssabs.append(rssab)
        
        self.conv=conv3x3(self.channels, self.channels)
        self.rssabs=nn.ModuleList(self.rssabs)
        
    def forward(self, x):
        
        for i, module in enumerate(self.rssabs):
            x=module(x) 
        
        x=self.conv(x)
        
        return x

    
    
    
