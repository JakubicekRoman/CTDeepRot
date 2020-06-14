import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import glob
import os
from skimage.io import imread
from skimage.transform import resize

from torch.utils import data
import os

from config import Config
import pandas as pd 

from skimage.transform import resize


class DataLoader(data.Dataset):
    
    def __init__(self, split,path_to_data):
    
        self.split=split
        self.path=path_to_data
        
        
        
        xl_file = pd.ExcelFile(self.path + os.sep +self.split+  os.sep+'labels.xlsx')

        data = pd.read_excel(xl_file,header=None)
        
        if self.split=='testing':
            data=data.loc[::11,:]
            
        
        
        file_names=data.loc[:,0].values.tolist()
        self.file_names=file_names
        
        labels=data.loc[:,1:4].to_numpy()
        self.labels=labels
        
        
        
    
    def __len__(self):
        return len(self.file_names)
    
    
    
    def __getitem__(self, index):
        
        file_name=self.file_names[index]
        
        img_list=[]
        
        folders = ['max_40','max_All','mean_20','mean_All','std_40','std_All']
        Rs= ['R3','R4','R1','R2','R5','R6']
        
        # folders = ['mean_All','std_All']
        # Rs= ['R2','R6']
        
        MEANS={'max_40': [0.3649016, 0.36678955, 0.3672551],
             'max_All': [0.41011006, 0.42946368, 0.43019485],
             'mean_20': [0.23211582, 0.23323949, 0.23327705],
             'mean_All': [0.34926993, 0.3492299, 0.34914622],
             'std_40': [0.15002619, 0.14761001, 0.14778571],
             'std_All': [0.2950393, 0.31131673, 0.31132057]}
        
        STDS={'max_40': [0.18549794, 0.17933437, 0.17979342],
             'max_All': [0.19559579, 0.1914579, 0.19143115],
             'mean_20': [0.15050271, 0.14773907, 0.14823063],
             'mean_All': [0.17176014, 0.16127543, 0.1615369],
             'std_40': [0.12240985, 0.11806898, 0.1184011],
             'std_All': [0.13937637, 0.13372615, 0.13382766]}
        
        for folder,R in zip(folders,Rs):
            
            for k in range(3):
                tmp=imread(self.path + os.sep +self.split +  os.sep +  folder +  os.sep + file_name  +'_' + R + '_Ch' + str(k+1)  + '.png')
                tmp=tmp.astype(np.float32)/255
                tmp=(tmp-MEANS[folder][k])/(STDS[folder][k])
                img_list.append(tmp)
                
                
                

        # if self.split=='training':
        #     resize_factor=0.2
            
            
    
            
        for k in range(len(img_list)):
            
            img_list[k]=resize(img_list[k],[224,224])
            
            
        if self.split=='training':
            
            max_mult_change=0.2
            for k in range(len(img_list)):
                mult_change=1+torch.rand(1).numpy()[0]*2*max_mult_change-max_mult_change
                img_list[k]=img_list[k]*mult_change
                
                
                
            max_add_change=0.1
            for k in range(len(img_list)):
                add_change=torch.rand(1).numpy()[0]*2*max_add_change-max_add_change
                img_list[k]=img_list[k]+add_change
            
        
        
            

            
  
        imgs=np.stack(img_list,axis=0)
        imgs=torch.from_numpy(imgs)
        
        
        
        lbl=self.labels[index,:]/180*np.pi
        
        Rx=np.array([[1,0,0],
                     [0,np.cos(lbl[0]),-np.sin(lbl[0])],
                     [0,np.sin(lbl[0]),np.cos(lbl[0])]])
        
        Ry=np.array([[np.cos(lbl[1]),0,np.sin(lbl[1])],
                     [0,1,0],
                     [-np.sin(lbl[1]),0,np.cos(lbl[1])]])
        
        Rz=np.array([[np.cos(lbl[2]),-np.sin(lbl[2]),0],
                     [np.sin(lbl[2]),np.cos(lbl[2]),0],
                     [0,0,1]])
        
        R=Rz@Ry@Rx
        lbl_vec=np.ones((3,1))
        
        lbl2=np.round(R@lbl_vec)[:,0]
        
        lbl2[lbl2==-1]=0
        
        lbl=torch.from_numpy(lbl2)
            

        return imgs,lbl
        
        
    
    
    
    
    
    

     