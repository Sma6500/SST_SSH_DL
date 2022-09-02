#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:10:41 2022

@author: lollier
"""


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Plot                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import os

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                           PLOT FUNCTIONS TO SHOW DATA AND RESULT                      | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


save_path="/usr/home/lollier/Documents/plots"

def imshow_area(array, fig_ax=None, vmin_vmax=None, title=False, save=False):
    """
    Parameters
    ----------
    array : numpy array
        array of shape (216,270) or (128,128) to plot
        if tensor (256,256), will use bicubic interpolation to force (216,270)
    title : str, optional
        title to plot. The default is False.
    save : str, optional
        saving the plot to the path given as input
        or to the plot file if True

    Returns
    -------
    plot the image
    """
    array, axes_label=check_shape(array)
    if fig_ax is None:
        fig, ax=plt.subplots(1,1)
        vmin, vmax=None, None
    else :
        fig, ax=fig_ax
        vmin, vmax= vmin_vmax
    
    img = ax.imshow(np.flipud(array), cmap='seismic', extent=axes_label, vmin=vmin, vmax=vmax)
    ax.set(xlabel='Longitude', ylabel='Latitude')
    ax.label_outer()

    if title:
        ax.set_title(title)
    fig.colorbar(img, ax=ax)
    if save:
        if type(save)==str:
            fig.savefig(os.path.join(save_path,save+'random_plot.png'))
        else :
            name=title if title else 'no_name'
            fig.savefig(os.path.join(save_path,name+'.png'))
    if ax is None:
        plt.show()


def check_shape(tensor):
    """
    Parameters
    ----------
    tensor : tensor or array
        check the size and the format of inputs and return the axes for plot.

    Returns
    -------
    tensor : TYPE
        DESCRIPTION.
    axes_label : TYPE
        DESCRIPTION.

    """
    
    axes_label=[-64.25,
                -41.833332,
                26.5,
                44.416668]
    
    if type(tensor)==torch.Tensor and tensor.shape==(256,256):
        t=transforms.Resize((216,270), interpolation=transforms.InterpolationMode.BICUBIC)
        tensor=torch.squeeze(t(tensor.view(1,1,256,256)))
        tensor=tensor.numpy()
        return tensor, axes_label
    
    elif type(tensor)==np.ndarray and tensor.shape==(216,270):
        return tensor, axes_label
    
    elif type(tensor)==np.ndarray and (tensor.shape==(128,128) or tensor.shape==(32,32)):
        axes_label=[-64.25,
                    -53.7,
                    26.5,
                    37.1]
        return tensor, axes_label
    
    else :
        raise ValueError('not tensor or shape not valid')
    
    
    
def plot_pred(target, last_time_step, prediction, title=None, save=False):
    """
    Parameters
    ----------
    target : TYPE
        DESCRIPTION.
    last_time_step : TYPE
        DESCRIPTION.
    prediction : TYPE
        DESCRIPTION.

    Returns
    -------
    plot the target vs the persistence and vs the prediction on similar scales.

    """
    
    if target.shape!=prediction.shape:
        print("target and prediction does not have the same shape : {},{}".format(target.shape, prediction.shape))
        return
    
    target, _=check_shape(target)
    last_time_step, _=check_shape(last_time_step)
    prediction, _=check_shape(prediction)
    
    fig, axs=plt.subplots(2,3)
    plt.figure(figsize=(30,20))
    
    #target and t-1
    vmin,vmax=np.min([target,last_time_step]),np.max([target,last_time_step])
    imshow_area(target, fig_ax=(fig,axs[0,0]), vmin_vmax=(vmin,vmax), title='target')
    imshow_area(last_time_step, fig_ax=(fig,axs[0,1]), vmin_vmax=(vmin,vmax), title='Input')
    
    #target and pred
    vmin,vmax=np.min([target,prediction]),np.max([target,prediction])
    imshow_area(target, fig_ax=(fig,axs[1,0]), vmin_vmax=(vmin,vmax), title='target')
    imshow_area(prediction, fig_ax=(fig,axs[1,1]), vmin_vmax=(vmin,vmax), title='Output (prédiction)')

    #pred and t-1
    v=np.max(np.absolute([target,last_time_step]))
    imshow_area(target-last_time_step, fig_ax=(fig,axs[0,2]), vmin_vmax=(-v,v), title='Difference')
    
    v=np.max(np.absolute([target,prediction]))
    imshow_area(target-prediction, fig_ax=(fig,axs[1,2]), vmin_vmax=(-v,v), title='Difference')
    
    if save:
        if type(save)==str:
            fig.savefig(os.path.join(save_path,save+'pred_plot.png'))
        else :
            name=title if title else 'no_name'
            fig.savefig(os.path.join(save_path,name+'.png'))
            
    plt.show()
    
    
    
def loss_plot(path, details=True, title=None, save=False):
    """
    Parameters
    ----------
    path : .txt str path
        Path to file that contains trainings print.
    details : bool
        if True plot the loss curves for each loss functions
    title : str
    
    Actions
    -------
    plot train and validation curve.
    """
    losses={'Train Loss':[], 
            'Validation Loss':[], 
            'Learning Rate':[0],
            'Learning Rate Epoch':[]}
    if details :
        losses_details={'Train  MSE_SLA':[],
                        'Valid  MSE_SLA':[],
                        'Train  MSE_SST':[],
                        'Valid  MSE_SST':[],
                        'Train  Grad_MSE':[],
                        'Valid  Grad_MSE':[]}

    with open(path, 'r') as f:
        data=f.readlines()
        for line in data:
            for key in losses.keys():
                if key in line : 
                    losses[key].append(float(line.split()[-1]))
            if details : 
                for key in losses_details.keys():
                    if key in line :
                        losses_details[key].append(float(line.split()[-1]))
        for i in range(len(losses['Learning Rate'])-1):
            if losses['Learning Rate'][i]!=losses['Learning Rate'][i+1]:
                losses['Learning Rate Epoch'].append(i)
        
    fig, ax=plt.subplots(1,1, figsize=(15,7))
    
    for key in losses.keys():
        if 'Loss' in key:
            ax.plot(np.log(losses[key]))
            
    ax.vlines(losses['Learning Rate Epoch'], 0, 1, transform=ax.get_xaxis_transform(), label='lr decreased', colors='r', alpha=0.1)
    
    if details : 
        for key in losses_details.keys():
            ax.plot(np.log(losses_details[key]))
    ax.legend(['Train Loss', 'Validation Loss'])# + [i for i in losses_details.keys()])
    if title is not(None): plt.title(title)
    if save:
        if type(save)==str:
            fig.savefig(os.path.join(save_path,save+'loss_plot.png'))
        else :
            name=title if title else 'no_name'
            fig.savefig(os.path.join(save_path,name+'.png'))
    plt.show()
    
    
if __name__=='__main__':
    
    # target=np.random.random((216,270))
    # imshow_area(target)
    # target=np.random.random((128,128))
    # imshow_area(target)
    # last_time_step=np.random.random((216,270))*5
    # prediction=np.random.random((216,270))
    
    # plot_pred(target,last_time_step, prediction)

    # target = torch.FloatTensor(np.random.random((256, 256)))
    # last_time_step = torch.FloatTensor(np.random.random((256, 256))*5)
    # prediction = torch.FloatTensor(np.random.random((256, 256)))

    # plot_pred(target, last_time_step, prediction)
    
    loss_plot("/usr/home/lollier/datatmp/weight_save/Unet/Unet_simple/overfit/overfit_next_time_step_6_.txt", title='Unet training and validation loss', details=False, save=False)
    

    
    

