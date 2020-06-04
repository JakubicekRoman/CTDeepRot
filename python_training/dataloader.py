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
        
        
        
        xl_file = pd.ExcelFile(self.path + os.sep +self.split+  os.sep+'labels_bin.xlsx')

        data = pd.read_excel(xl_file,header=None)
        
        file_names=data.loc[:,0].values.tolist()
        self.file_names=file_names
        
        self.labels=data.loc[:,1:7].to_numpy()
        
        
        
    
    def __len__(self):
        return len(self.file_names)
    
    
    
    def __getitem__(self, index):
        
        file_name=self.file_names[index]
        
        img_list=[]
        
        folders = ['max_40','max_All','mean_20','mean_All','std_40','std_All']
        Rs= ['R3','R4','R1','R2','R5','R6']
        
        for folder,R in zip(folders,Rs):
            
            for k in range(3):
                img_list.append(imread(self.path + os.sep +self.split +  os.sep +  folder +  os.sep + file_name  +'_' + R + '_Ch' + str(k+1)  + '.png'))
            
            
         
            
        for k in range(len(img_list)):
            
            img_list[k]=resize(img_list[k].astype(np.float32),[224,224])
            
        
            
        imgs=np.stack(img_list,axis=0) -0.5
        imgs=torch.from_numpy(imgs)
        
        lbl=self.labels[index,:]
        lbl=torch.from_numpy(lbl)
            
        
        
        return imgs,lbl
        
        
    
    
    
    
    
    

     