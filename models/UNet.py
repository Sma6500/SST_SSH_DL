#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:06:32 2022

@author: lollier

largely inspired from : 
https://github.com/jaxony/unet-pytorch

Changes : 
    
    - changes zero padding to replicate padding 
    - add batch normalisation
    - add the hyperparameter SiLU or ReLU for activation function
    - create an output block adding two convolutions
    - add the possibilitie to add to the output the last input in order to predict a transformation mask instead of a field
    
"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         UNET                                          | #
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

"""
UNet basic implementation
"""

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
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
        return nn.ConvTranspose2d(
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
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def batch_norm(in_channels):
    return nn.BatchNorm2d(in_channels)

class ScaleLayer(nn.Module):

   def __init__(self, init_value=1e-1):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, x):
       return x * self.scale

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
        self.norm = batch_norm(out_channels)
        self.activation=activation

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x=self.conv1(x)
        x=self.norm(x)
        x=self.activation(x)
        #x = self.activation(self.norm(self.conv1(x)))
        x = self.activation(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU/SiLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose', activation=nn.ReLU()):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.norm = batch_norm(out_channels)
        self.activation=activation


        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = self.activation(self.conv1(x))
        x = self.norm(self.activation(self.conv2(x)))
        #x = self.activation(self.conv2(x))
        return x
        
    
class Output_Block(nn.Module):
    """
    CURRENTLY NOT USED
    A postprocessing module that performs a Batch normalization and 2 convolutions (with padding).
    """
    def __init__(self, in_channels, out_channels, n_hidden=32):
        
        super(Output_Block, self).__init__()
        #self.norm=batch_norm(in_channels)
        self.conv=nn.Sequential(conv3x3(in_channels, n_hidden),
                                conv3x3(n_hidden, out_channels))
        
    def forward(self, x):
                
        #out=self.norm(x)
        out=self.conv(x)
        
        return out
        
                                

class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    (5) input Block that apply two convolutionnal filters and a batchnormalisation
        on data. Add to its output the oldest input of the timeseries
    """

    def __init__(self, num_classes, in_channels=8, depth=5,
                 start_filts=64, 
                 up_mode='transpose', merge_mode='concat', activation='ReLU',
                 transfo_field=True, nb_inputs=2):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 8 for 4 sla and 4 sst images.
            first_block_channels : int, number of convolutionnal filters for 
                the input processing block
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.,
            activation : string ReLU or SiLU for activation function
            transfo_field : False for simple Unet, True for adding last input at the end of the forward
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
            
        if activation in ('ReLU', 'SiLU'):
            if activation=='ReLU' :
                self.activation = nn.ReLU()
            else : 
                self.activation = nn.SiLU()
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "activation. Only \"SiLU\" and "
                             "\"ReLU\" are allowed.".format(activation))
            
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.nb_inputs=nb_inputs
        self.norm=batch_norm(in_channels)
        self.down_convs = []
        self.up_convs = []
        self.transfo_field=transfo_field
        self.pred_sst=True if self.num_classes>1 else False

        
        # if self.transfo_field : 
        #     self.last_block=Output_Block(num_classes, num_classes)


        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling, activation=self.activation)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode, activation=self.activation)
            self.up_convs.append(up_conv)
            
        #self.output_block=Output_Block(self.start_filts, 16)
        self.conv_final = conv1x1(self.start_filts, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        
        if self.transfo_field:
            last_sla_pos=x.shape[1]//2-1

            last_sla=x[:, last_sla_pos,:,:]
            
            if self.pred_sst:
                last_sst=x[:,-1,:,:]
                last_input=torch.stack([last_sla,last_sst], axis=1)
            else :
                last_input=torch.unsqueeze(last_sla, axis=1)
            
        
        encoder_outs = []         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        
        #x=self.output_block(x)
        x = self.conv_final(x)
        if self.transfo_field:
                # x=self.last_block(x+torch.unsqueeze(last_sla, 1))
                x=x+last_input
        
        return x

if __name__ == "__main__":
    """
    testing
    """
    #from torchsummary import summary
    model = UNet(2, depth=5, merge_mode='concat', activation='ReLU', transfo_field=False)
    model.to(device='cuda')
    #block_input=Input_Block(8, 32)
    #summary(model, input_size=(8,128,128))
    
    # input_names = ['Sentence']
    # output_names = ['yhat']
    # a=torch.Tensor(np.random.random(((32,8,128,128)))).to(device='cuda')
    # torch.onnx.export(model, a, 'test0.onnx', input_names=input_names, output_names=output_names)
