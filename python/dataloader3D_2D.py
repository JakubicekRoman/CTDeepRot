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

from scipy.ndimage import zoom


from utils.rotate_fcns import rotate_2d,rotate_3d,flip_2d



class DataLoader3D_2D(data.Dataset):
    
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
            
            file_names.append((self.path + os.sep + folder.split('\\')[-1] + os.sep + name).replace('.mhd','.npy'))
        
        
            

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

                            
          
        q=1
                  
        
    
    def __len__(self):
        return len(self.file_names)
    
    
    
    def __getitem__(self, index):
        
        file_name=self.file_names[index]
        
        
        file_name_3d=file_name
        
        file_name_3d=file_name_3d.replace('CT_rotation_data_2D','CT_rotation_data_npy_128')
        
        
        file_name_2d=file_name[:-4]
        
        MEAN=614.2868
        STD=614.2868
        img=np.load(file_name_3d)
        img=img.astype(np.float32)
        img=(img-MEAN)/STD
        
            


        r=self.vec[index][0:3]
        flip=self.flip[index]
        flip=np.array([flip])
            
        if flip:
            img=img[:,::-1,:]
            
            
        img=rotate_3d(img,r)
        
        img=np.expand_dims(img, axis=0).copy()
        img3d=torch.from_numpy(img)

        
        img_list=[]
        
        folders=['mean','max','std']
        
        for folder in folders:
            
            for k in range(3):
                tmp=imread(file_name_2d + '_' + folder + '_'+ str(k+1)  +'.png' )
                tmp=tmp.astype(np.float32)/255-0.5
                img_list.append(tmp)
        
        
        
        imgs=np.stack(img_list,axis=2)
        for k in range(0,9,3):
            if flip==1:
                imgs[:,:,k:k+3]=flip_2d(imgs[:,:,k:k+3])
            
            imgs[:,:,k:k+3]=rotate_2d(imgs[:,:,k:k+3],r)
        


        imgs=torch.from_numpy(imgs.copy())
        
        img2d=imgs.permute(2,0,1)
        
        
        
        lbl=self.lbls[index]
        lbl2=np.zeros(self.rots_table.shape[0]).astype(np.float32)
        lbl2[lbl]=1
        
        
        lbl=torch.from_numpy(lbl2)
            
        return img3d,img2d,lbl
        
        
    
    
    
    
    
    

     