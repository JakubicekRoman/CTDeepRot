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
        self.test_flip=np.random.randint(0, 2, size=(len(self.file_names),1))
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
            flip=torch.randint(2,(1,1)).view(-1).numpy()
            



        elif self.split=='testing':
            a,b,c=self.test_vec[index,:]
            flip=self.test_flip[index,0]
            a=np.array([a])
            b=np.array([b])
            c=np.array([c])
            flip=np.array([flip])
            
        if flip:
            img=img[:,:,::-1]
            
        # plt.imshow(np.max(img,axis=0))
        # plt.show()
        # plt.imshow(np.max(img,axis=1))
        # plt.show()
        # plt.imshow(np.max(img,axis=2))
        # plt.show()
            
        img=np.rot90(img,a,axes=(0,1))
        img=np.rot90(img,b,axes=(0,2))
        img=np.rot90(img,c,axes=(1,2))
        
        if self.split=='training':
            max_mult_change=0.2
            mult_change=1+torch.rand(1).numpy()[0]*2*max_mult_change-max_mult_change
            img=img*mult_change
            
                
            max_add_change=0.2
            add_change=torch.rand(1).numpy()[0]*2*max_add_change-max_add_change
            img=img+add_change
            
            
            max_cicrcshift_change=20
            cicrcshift_change=torch.randint(2*max_cicrcshift_change,(3,1)).view(-1).numpy()-max_cicrcshift_change
            img=np.roll(img,cicrcshift_change[0],axis=0)
            img=np.roll(img,cicrcshift_change[1],axis=1)
            img=np.roll(img,cicrcshift_change[2],axis=2)
            
            # shape=img.shape
            # max_resize_change=0.3
            # max_resize_change=np.round(max_resize_change*shape)
            # shape_new=shape*torch.rand(3).numpy()[0]*2*max_resize_change-max_resize_change
            # new_img=np.zeros(shape_new,dtype=np.float32)
            # min_shape=np.array([np.min([shape[0],shape_new[0]]),np.min([shape[1],shape_new[1]]),np.min([shape[2],shape_new[2]])] )
            # new_img[]
            

        
        lbls_angle=np.concatenate((a,b,c))*90

        
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
        
        
    
    
    
    
    
    

     