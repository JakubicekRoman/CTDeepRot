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


from utils.rotate_fcns import rotate_2d,rotate_3d,flip_2d


class DataLoader2D(data.Dataset):
    
    def __init__(self, split,path_to_data):
    
        self.split=split
        self.path=path_to_data
        
        data = pd.read_csv("utils/rot_dict_unique.csv") 
        self.rots_table=data.loc[:,:].to_numpy()
        
        xl_file = pd.ExcelFile(self.path + os.sep+'ListOfData.xlsx')
        data = pd.read_excel(xl_file,header=None)
        
        folders=data.loc[:,0].tolist()
        names=data.loc[:,1].tolist()
        file_names=[]
        for folder,name in zip(folders,names):
            
            file_names.append((self.path + os.sep + folder.split('\\')[-1] + os.sep + name).replace('.mhd',''))
        
        
            

        if self.split=='training':
            file_names=file_names[:int(len(file_names)*0.8)]
            
        elif self.split=='testing':
            file_names=file_names[int(len(file_names)*0.8):-20]
            
        
        self.file_names=[]
        self.vec=[]
        self.flip=[]
        self.lbls=[]
        for file in file_names:
            for flip in [0,1]:
                for unique_rot_num in range(self.rots_table.shape[0]):
                    
                    self.file_names.append(file)
                    self.vec.append(self.rots_table[unique_rot_num,:])
                    self.flip.append(flip)
                    self.lbls.append(unique_rot_num)
        
        
        
    
    def __len__(self):
        return len(self.file_names)
    
    
    
    def __getitem__(self, index):
        
        file_name=self.file_names[index]
        
        r=self.vec[index][0:3]
        flip=self.flip[index]
        flip=np.array([flip])
        
        img_list=[]
        

        
        MEANS={'mean': [0.31656316, 0.31815434, 0.319901],
             'max': [0.48267424, 0.38830274, 0.37235856],
             'std': [0.40330338, 0.29134238, 0.23324046]}
        
        STDS={'mean': [0.11143641, 0.18203282, 0.20284137],
             'max': [0.15405223, 0.2034344, 0.22770199],
             'std': [0.11865007, 0.16135372, 0.15640634]}
        
        folders=['mean','max','std']
        
        for folder in folders:
            
            for k in range(3):
                tmp=imread(file_name + '_' + folder + '_'+ str(k+1)  +'.png' )
                tmp=tmp.astype(np.float32)/255
                tmp=(tmp-MEANS[folder][k])/(STDS[folder][k])
                img_list.append(tmp)
                

            
        # if self.split=='training':
            
        #     max_mult_change=0.3
        #     for k in range(len(img_list)):
        #         mult_change=1+torch.rand(1).numpy()[0]*2*max_mult_change-max_mult_change
        #         img_list[k]=img_list[k]*mult_change
                
                
                
        #     max_add_change=0.3
        #     for k in range(len(img_list)):
        #         add_change=torch.rand(1).numpy()[0]*2*max_add_change-max_add_change
        #         img_list[k]=img_list[k]+add_change
            
        
        
            

            
  
        imgs=np.stack(img_list,axis=2)
        for k in range(0,9,3):
            if flip==1:
                imgs[:,:,k:k+3]=flip_2d(imgs[:,:,k:k+3])
            
            imgs[:,:,k:k+3]=rotate_2d(imgs[:,:,k:k+3],r)
        


        imgs=torch.from_numpy(imgs.copy())
        
        imgs=imgs.permute(2,0,1)

        
        lbl=self.lbls[index]
        lbl2=np.zeros(self.rots_table.shape[0]).astype(np.float32)
        lbl2[lbl]=1
        
        lbl=torch.from_numpy(lbl2)
            

        return imgs,lbl
        
        
    
    
    
    
    
    

     