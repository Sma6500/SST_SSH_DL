#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:35:23 2022

@author: lollier


"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         DATALOADERS                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from utils.functions import minmax_scaler

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
"""
Custom Dataset inheriting from Pytorch Dataset
Take a .npy image time series as dataset and create time series for training of lenght L_source+L_target
"""

class time_serie_dataset(Dataset):

    def __init__(self, datasets, L_source, L_target, dt, transform=None):
        """
        Parameters
        ----------
        dataset : numpy
            numpy time series dataset (shape : time, *image_shape)
        L_source, L_target : int
            lenghts of network input source and output target
        dt : int
            timestep for the time series in days
        transform : pytorch transform, optional
            transformation to apply to the data. The default is None.
        """
        
        self.dataset_sla, self.dataset_sst = datasets
        self.transform = transform
        self.L_source = L_source*dt
        self.L_target = L_target*dt
        self.dt = dt


    def __len__(self):
        return len(self.dataset_sla)-(self.L_source+self.L_target)
    
    def __getitem__(self, index):
        """
        Main function of the CustomDataset class. 
        """
        item={}
        item['source_sla'], item['target_sla'] = load(index, self.dataset_sla, self.L_source, self.L_target, self.dt)
        item['source_sst'], item['target_sst'] = load(index, self.dataset_sst, self.L_source, self.L_target, self.dt)
        
        if self.transform is not None:
            
            item['source_sla'], item['target_sla'] = self.transform(np.moveaxis(item['source_sla'], 0,-1)),self.transform(np.moveaxis(item['target_sla'], 0,-1))
            item['source_sst'], item['target_sst'] = self.transform(np.moveaxis(item['source_sst'], 0,-1)),self.transform(np.moveaxis(item['target_sst'], 0,-1))
            
        return item

def load(index, dataset, L_source, L_target, dt):
    """
    Parameters
    ----------
    index : int
    dataset : numpy
        numpy time series dataset (shape : time, *image_shape)
    L_source, L_target : int
        lenght in days of network input source and output target
    dt : int
        timestep for the time series in days

    Returns
    -------
    source : numpy
        time series of shape (L_source/dt, *image_shape).
    target : numpy
        time series of shape (L_target/dt, *image_shape).
    """
    
    index_source=[i for i in range(index, index+L_source, dt)]
    index_target=[i for i in range(index+L_source, index+L_source+L_target, dt)]
    
    source = np.stack([dataset[i] for i in index_source])
    target = np.stack([dataset[i] for i in index_target])
    
    return source, target


def get_dataloaders(dataloader_config):
    """
    Parameters
    ----------
    dataloader_config : dict
        dataloader configuration (see config.py).

    Returns
    -------
    Pytorch Dataloader or dict of Dataloader 
    """
        
    datasets=[np.load(i) for i in dataloader_config['dataset_path']]
    if dataloader_config['normalize']:
        datasets=[minmax_scaler(data) for data in datasets]
    
    temp=int(dataloader_config['valid_ratio']*len(datasets[0]))
    training_set, validation_set = (datasets[0][:-temp], datasets[1][:-temp]),(datasets[0][-temp:], datasets[1][-temp:])
    
    training_set=time_serie_dataset(training_set, 
                                    dataloader_config['lenght_source'],
                                    dataloader_config['lenght_target'],
                                    dataloader_config['timestep'],
                                    dataloader_config['transform'])
    
    validation_set=time_serie_dataset(validation_set, 
                                    dataloader_config['lenght_source'],
                                    dataloader_config['lenght_target'],
                                    dataloader_config['timestep'],
                                    dataloader_config['transform'])
    

    training_generator   = DataLoader(training_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True)
    validation_generator = DataLoader(validation_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=False)
    
    if dataloader_config['small_train']:
        train_sampler = SubsetRandomSampler([i for i in range(0,len(training_set),2)]) #only half data to compute faster 
        training_generator   = DataLoader(training_set,
                                          batch_size=dataloader_config['batch_size'],
                                          sampler=train_sampler)
        
    return training_generator, validation_generator    
    

if __name__=='__main__': 
    
    # from torchvision import transforms
    # from torchvision.transforms.functional import crop
    # def crop128(image):
    #     return crop(image, 0, 0, 128, 128)
    
    # dataloader_config={'dataset_path':["/usr/home/lollier/datatmp/npy_data/train/sla_ts_residus.npy","/usr/home/lollier/datatmp/npy_data/train/sst_ts_residus.npy"],
    #                    'lenght_source': 6,
    #                    'lenght_target': 6,
    #                    'timestep':5,
    #                    'transform':transforms.Compose([transforms.ToTensor(), transforms.Lambda(crop128)]),
    #                    'valid_ratio':0.2,
    #                    'batch_size': 32,
    #                    'normalize':True,
    #                    'small_train':True}
    
    # test=get_dataloaders(dataloader_config)
    
    a=[i for i in range(10)] 
    for i in a :
        a.append(1)
        print(i)
        
        
        
        
        
        
        
        
        
        
    
    
    
    