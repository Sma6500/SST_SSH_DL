#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:19:37 2022

@author: lollier
"""

import torch
import numpy as np
import os
from utils.plot import *
from torchvision import transforms
from utils.functions import *
from Dataloaders import time_serie_dataset
from models.UNet import UNet
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import matplotlib.pyplot as plt
from model import model

from tqdm import tqdm
#plot loss training

path_train='/usr/home/lollier/datatmp/weight_save/Unet/Unet_simple/'



path_data_1993=["/usr/home/lollier/datatmp/npy_data/test/sla_ts_residus_1993.npy","/usr/home/lollier/datatmp/npy_data/test/sst_ts_residus_smooth_1993.npy"]
path_data_2019=["/usr/home/lollier/datatmp/npy_data/test/sla_ts_residus_smooth_2019.npy","/usr/home/lollier/datatmp/npy_data/test/sst_ts_residus_smooth_2019.npy"]
path_data=["/usr/home/lollier/datatmp/npy_data/sla_ts_residus.npy","/usr/home/lollier/datatmp/npy_data/sst_ts_residus.npy"]

path_sla_not_normed="/usr/home/lollier/datatmp/npy_data/sla_ts_residus.npy"
path_sst_not_normed="/usr/home/lollier/datatmp/npy_data/sst_ts_residus.npy"

save_path="/usr/home/lollier/Documents/plots"

class Tester():
    
    def __init__(self, path_test_data, path, net_name, path_weight_load=None, path_config=None, range_pred=None, save=False):
        
        if path_config is None:
            self.path_config=os.path.join(path, net_name+'_config.json')
        else : 
            self.path_config=path_config
            
        if path_weight_load is None:
            self.path_weight_load=os.path.join(path, net_name+'best_valid_loss.pt')
        else : 
            self.path_weight_load=path_weight_load
            
        self.save=save
        self.path=path
        self.net_name=net_name
        
        self.__get_config__(self.path_config)
        
        if range_pred is not(None) :
            self.dataloader_config['lenght_target']=range_pred
            
        print("building model\n")
        net=UNet(self.unet_config['num_predictions'], in_channels=self.dataloader_config['lenght_source']*2,
                 depth=self.unet_config['depth'],start_filts=self.unet_config['start_filts'], merge_mode=self.unet_config['merge_mode'], activation=self.unet_config['activation'], transfo_field=self.unet_config['transfo_field'])
        self.model = model(net, self.criterion_config, self.optimizer_config, self.scheduler_config) #from class Model
        self.model.net.load_state_dict(torch.load(self.path_weight_load, map_location=self.model.device))
        
        self.pred_sst=True if net.num_classes>1 else False

        self.__load_data__(path_test_data)
        self.prediction_rmse,self.persistence_rmse=self.test_step()
        
    def __get_config__(self, path_config):
        """
        Parameters
        ----------
        path_config : str
            path to the json file containing the config dict

        Action
        -------
        load all dict configs in self variable
        """
        
        t=transforms.Compose([transforms.ToTensor(), transforms.Lambda(crop128)])
        self.unet_config, self.dataloader_config, self.train_config, self.criterion_config, self.scheduler_config, self.optimizer_config = load_config(path_config, t)
        self.dataloader_config['lenght_target']=self.train_config['prediction_probabilities'][2] if self.dataloader_config['lenght_target'] is None else self.dataloader_config['lenght_target']
        self.criterion_config.pop('MSE',None)
        self.criterion_config['RMSE']=1.
        self.criterion_config.pop('Grad_MSE',None)
        self.criterion_config['details']=False
    
    def __load_data__(self,path_test_data):
        """
        Parameters
        ----------
        path_test_data : str

        Actions
        -------
        build dataset and dataloader
        """
        print("data loading\n")
        datasets=[np.load(i) for i in path_test_data]
        if not('normalize' in self.dataloader_config.keys()) or self.dataloader_config['normalize']:
            datasets=[minmax_scaler(data) for data in datasets]
            
            #pourquoi je fais ça là ? j'ai déjà normalisé ducoup ça marche pas non ?
            self.max_sla=np.max(datasets[0])
            self.max_sst=np.max(datasets[1])

            self.min_sla=np.min(datasets[0])
            self.min_sst=np.min(datasets[1])
            
        self.test_set=time_serie_dataset(datasets, 
                                        self.dataloader_config['lenght_source'],
                                        self.dataloader_config['lenght_target'],
                                        self.dataloader_config['timestep'],
                                        self.dataloader_config['transform'])
        sampler=SequentialSampler(self.test_set)
        
        self.test_loader=DataLoader(self.test_set, sampler=sampler)    
        
    def denormalize(self, data, cat='sla'):
        """
        Parameters
        ----------
        data : torch tensor (128,128)
            image to unnormalize.

        Raises
        ------
        ValueError
            if shape or type is not correct

        Returns
        -------
        data : torch tensor (128,128)
            image without normalization
        """
        
        data=torch.squeeze(data)
        if type(data)!=torch.Tensor and data.shape!=(128,128):
            raise ValueError("unvalid shape and or unvalid type")
        
        if cat=='sla':
            data=(data*(self.max_sla-self.min_sla))+self.min_sla
        elif cat=='sst':
            data=(data*(self.max_sst-self.min_sst))+self.min_sst
        else :
            raise ValueError("unvalid cat, needs sst or sla")
        return data


    def test_step(self):
        """
        Returns
        -------
        pred_rmse : dict
            rmse for each prediction range on test set.
        persistence_rmse : dict
            rmse for each persistence range on test set.
        """
        print("\ncomputing rmse\n")
        pred_rmse={}
        persistence_rmse={}
        for i in range(self.dataloader_config['lenght_target']):
            pred_rmse["rmse_sla_pred_t+{}".format(self.dataloader_config['timestep']*(i+1))]=[]
            persistence_rmse["rmse_sla_persi_t+{}".format(self.dataloader_config['timestep']*(i+1))]=[]
            
            if self.pred_sst:
                pred_rmse["rmse_sst_pred_t+{}".format(self.dataloader_config['timestep']*(i+1))]=[]
                persistence_rmse["rmse_sst_persi_t+{}".format(self.dataloader_config['timestep']*(i+1))]=[]


        self.model.net.eval()
        with torch.no_grad():
            for item in tqdm(self.test_loader):

                prediction = self.model.net(torch.cat([item['source_sla'], 
                                                       item['source_sst']], 
                                                      axis=1).to(self.model.device))
                #print('source{}'.format(item['source_sla'][0,-1,5,5]))
                # print(f'prediction{prediction[0,0,5,5]}')
                # print('target{}'.format(item['target_sla'][0,0,5,5]))
                #stock pred and pers
                pred_rmse["rmse_sla_pred_t+{}".format(self.dataloader_config['timestep'])].append(self.model.criterion(self.denormalize(prediction[:,0,:,:]), 
                                                                                                                       self.denormalize(item['target_sla'][:,0,:,:]).to(self.model.device)).item())  
                pred_rmse["rmse_sst_pred_t+{}".format(self.dataloader_config['timestep'])].append(self.model.criterion(self.denormalize(prediction[:,1,:,:],cat='sst'), 
                                                                                                                       self.denormalize(item['target_sst'][:,0,:,:], cat='sst').to(self.model.device)).item())  
                persistence_rmse["rmse_sla_persi_t+{}".format(self.dataloader_config['timestep'])].append(self.model.criterion(self.denormalize(item['source_sla'][:,-1,:,:].to(self.model.device)), 
                                                                                                                               self.denormalize(item['target_sla'][:,0,:,:]).to(self.model.device)).item())
                persistence_rmse["rmse_sst_persi_t+{}".format(self.dataloader_config['timestep'])].append(self.model.criterion(self.denormalize(item['source_sst'][:,-1,:,:].to(self.model.device),cat='sst'), 
                                                                                                                               self.denormalize(item['target_sst'][:,0,:,:],cat='sst').to(self.model.device)).item())
                
                mem_pred_sla=torch.unsqueeze(torch.clone(prediction[:,0,:,:]),1)
                mem_pred_sst=torch.unsqueeze(torch.clone(prediction[:,1,:,:]),1)
                
                for i in range(self.dataloader_config['lenght_target']-1):
                    
                    prediction=self.model.net(torch.cat([item['source_sla'][:,i+1:,:,:].to(self.model.device),
                                                          mem_pred_sla,
                                                            item['source_sst'][:,i+1:,:,:].to(self.model.device)
                                                            , mem_pred_sst], axis=1))
                    
                    mem_pred_sla=torch.cat([mem_pred_sla,torch.unsqueeze(prediction[:,0,:,:],1)], axis=1)
                    mem_pred_sst=torch.cat([mem_pred_sst,torch.unsqueeze(prediction[:,1,:,:],1)], axis=1)
                    
                    pred_rmse["rmse_sla_pred_t+{}".format(self.dataloader_config['timestep']*(i+2))].append(self.model.criterion(self.denormalize(prediction[:,0,:,:]), 
                                                                                                                                 self.denormalize(item['target_sla'][:,i+1,:,:]).to(self.model.device)).item())
                    pred_rmse["rmse_sst_pred_t+{}".format(self.dataloader_config['timestep']*(i+2))].append(self.model.criterion(self.denormalize(prediction[:,1,:,:],cat='sst'), 
                                                                                                                                 self.denormalize(item['target_sst'][:,i+1,:,:],cat='sst').to(self.model.device)).item())
                    persistence_rmse["rmse_sla_persi_t+{}".format(self.dataloader_config['timestep']*(i+2))].append(self.model.criterion(self.denormalize(item['source_sla'][:,-1,:,:].to(self.model.device)), 
                                                                                                                    self.denormalize(item['target_sla'][:,i+1,:,:]).to(self.model.device)).item())
                    persistence_rmse["rmse_sst_persi_t+{}".format(self.dataloader_config['timestep']*(i+2))].append(self.model.criterion(self.denormalize(item['source_sst'][:,-1,:,:].to(self.model.device),cat='sst'), 
                                                                                                                    self.denormalize(item['target_sst'][:,i+1,:,:],cat='sst').to(self.model.device)).item())
        return pred_rmse, persistence_rmse
    
    def loss_plot(self):
        
        loss_plot(os.path.join(self.path,self.net_name+'.txt'), save=self.save)
        
    def rmse_plot(self):
        
        for i,j in zip(self.prediction_rmse.keys(),self.persistence_rmse.keys()):
            
            plt.plot([b-a for a,b in zip(self.prediction_rmse[i],self.persistence_rmse[j])])
            plt.axhline(0, c='r', alpha=0.5)
            plt.title('RMSE '+j[5:]+' - '+i[5:])
            if self.save:
                if type(self.save)==str:
                    plt.savefig(os.path.join(save_path,self.save+'rmse_plot'+j[5:]+' - '+i[5:]+'.png'))
                else :
                    name='RMSE '+j[5:]+' - '+i[5:]
                    fig.savefig(os.path.join(save_path,name+'.png'))
                    
            plt.show()
    
    def visualize_sample(self, index, scale=True, range_pred=0):
        
        item=self.test_set[index]
        prediction=self.model.net(torch.unsqueeze(torch.cat([item['source_sla'].to(self.model.device),
                                                             item['source_sst'].to(self.model.device)], axis=0),0))
        mem_pred_sla=torch.unsqueeze(torch.clone(prediction[:,0,:,:]),1)
        mem_pred_sst=torch.unsqueeze(torch.clone(prediction[:,1,:,:]),1)
        for i in range(range_pred):
            prediction=self.model.net(torch.cat([torch.unsqueeze(item['source_sla'][i+1:,:,:].to(self.model.device),0),
                                                 mem_pred_sla,
                                                 torch.unsqueeze(item['source_sst'][i+1:,:,:].to(self.model.device),0),
                                                 mem_pred_sst], axis=1))        
            mem_pred_sla=torch.cat([mem_pred_sla,torch.unsqueeze(prediction[:,0,:,:],1)], axis=1)
            mem_pred_sst=torch.cat([mem_pred_sst,torch.unsqueeze(prediction[:,1,:,:],1)], axis=1)
        if scale:
            plot_pred(self.denormalize(self.test_set[index]['target_sla'][range_pred,:,:].cpu()).numpy(), self.denormalize(self.test_set[index]['source_sla'][-1,:,:].cpu()).numpy(), self.denormalize(prediction[0,0,:,:].cpu()).detach().numpy(), save=self.save)
            
            plot_pred(self.denormalize(self.test_set[index]['target_sst'][range_pred,:,:].cpu(), cat='sst').numpy(), self.denormalize(self.test_set[index]['source_sst'][-1,:,:].cpu(), cat='sst').numpy(), self.denormalize(prediction[0,1,:,:].cpu(),cat='sst').detach().numpy(), save=self.save)
        
        else : 
            imshow_area(self.denormalize(self.test_set[index]['target_sla'][range_pred,:,:].cpu()).numpy(), title='t-1', save=self.save)
            imshow_area(self.denormalize(self.test_set[index]['source_sla'][-1,:,:].cpu()).numpy(), title='target', save=self.save)
            imshow_area(self.denormalize(prediction[0,0,:,:].cpu()).detach().numpy(), title='pred')

            imshow_area(self.denormalize(self.test_set[index]['target_sst'][range_pred,:,:].cpu()).numpy(), title='t-1',save=self.save)
            imshow_area(self.denormalize(self.test_set[index]['source_sst'][-1,:,:].cpu()).numpy(), title='target',save=self.save)
            imshow_area(self.denormalize(prediction[0,1,:,:].cpu()).detach().numpy(), title='pred',save=self.save)
    
if __name__=="__main__":
    
    
    
    import os 
    path_data_1993=["/usr/home/lollier/datatmp/npy_data/test/sla_ts_residus_1993.npy","/usr/home/lollier/datatmp/npy_data/test/sst_ts_residus_1993.npy"]
    path_data_2019=["/usr/home/lollier/datatmp/npy_data/test/sla_ts_residus_2019.npy","/usr/home/lollier/datatmp/npy_data/test/sst_ts_residus_2019.npy"]
    path_ablation="/usr/home/lollier/datatmp/weight_save/Unet/Ablation/"
    weights=os.listdir(path_ablation)
    for weight in weights:
        if weight[-18:]=='best_valid_loss.pt':
            net_name=weight[:-18]
            print(net_name)
            test_1993=Tester(path_data_1993, path_ablation, net_name, path_config=os.path.join(path_ablation, net_name+'_config.json'), range_pred=3, save=net_name+'_1993')#,path_config=os.path.join(path_train, 'overfit_next_time_step_6__config.json'))
            test_2019=Tester(path_data_2019, path_ablation, net_name, path_config=os.path.join(path_ablation, net_name+'_config.json'), range_pred=3, save=net_name+'_2019')#,path_config=os.path.join(path_train, 'overfit_next_time_step_6__config.json'))
            
            test_1993.loss_plot()
            test_2019.loss_plot()

            test_1993.rmse_plot()
            test_2019.rmse_plot()

            test_1993.visualize_sample(200, scale=True)
            test_2019.visualize_sample(200, scale=True)
            
    # def a(tester,i):
    #     for k,j in zip(tester.prediction_rmse.keys(),tester.persistence_rmse.keys()):
            
    #         plt.plot(tester.prediction_rmse[k])
    #         plt.plot(tester.persistence_rmse[j])
    #         plt.legend(['predict','persist'])
    
    #         plt.show()
        
        
        
    # test_all.loss_plot()
    # test_all.rmse_plot()
    # test_all.visualize_sample(2500, scale=True)
    # test_all.visualize_sample(-300, scale=True)

    # for i in range(5):
    #     test_all.visualize_sample(2500, scale=True, range_pred=i)
    
    
    
    #test_all=Tester(path_data, path_train, 'overfit_next_time_step_2_', path_weight_load=os.path.join(path_train,'checkpoint_400overfit_next_time_step_2__.pt'), range_pred=2)


























