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
        
        
        
        if self.split=='testing':
            self.file_names=glob.glob(self.path + os.sep + 'Data_raw_test' + os.sep + '*.npy', recursive=True)

        elif self.split=='training':
            self.file_names=glob.glob(self.path + os.sep + 'Data_raw_train' + os.sep + '*.npy', recursive=True)

        
        
        self.file_names=self.file_names*4
        
        state=np.random.get_state()
        np.random.seed(42)
        self.test_vec=np.random.randint(0, 4, size=(len(self.file_names),3))
        np.random.set_state(state)
        
    
    def __len__(self):
        return len(self.file_names)
    
    
    
    def __getitem__(self, index):
        
        file_name=self.file_names[index]
        
        MEAN=900.3071
        STD=318.11615
        img=np.load(file_name)
        img=img.astype(np.float32)
        img=(img-MEAN)/STD
     
        if self.split=='training':
            a=torch.randint(4,(1,1)).view(-1).numpy()
            b=torch.randint(4,(1,1)).view(-1).numpy()
            c=torch.randint(4,(1,1)).view(-1).numpy()
            
            



        elif self.split=='testing':
            a,b,c=self.test_vec[index,:]
        
            
    
        img=np.rot90(img,a,axes=(0,1))
        img=np.rot90(img,b,axes=(0,2))
        img=np.rot90(img,c,axes=(1,2))
        
        lbls_angle=np.concatenate((a,b,c))*180

        
        img=np.expand_dims(img, axis=0).copy()
        img=torch.from_numpy(img)
        
        lbl=lbls_angle/180*np.pi
        
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
            

        return img,lbl
        
        
    
    
    
    
    
    

     