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

from utils.rotate_fcns import rotate_2d,rotate_3d



class DataLoader3D(data.Dataset):
    
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
        
        MEAN=614.2868
        STD=614.2868
        img=np.load(file_name)
        img=img.astype(np.float32)
        img=(img-MEAN)/STD
        
            


        r=self.vec[index][0:3]
        flip=self.flip[index]
        flip=np.array([flip])
            
        if flip:
            img=img[:,::-1,:]
            
            
        img=rotate_3d(img,r)
        
        # if self.split=='training':
            
            
        #     max_mult_change=0.3
        #     mult_change=1+torch.rand(1).numpy()[0]*2*max_mult_change-max_mult_change
        #     img=img*mult_change
            
            
            
                
        #     max_add_change=0.3
        #     add_change=torch.rand(1).numpy()[0]*2*max_add_change-max_add_change
        #     img=img+add_change
            
            
            
            
        #     max_cicrcshift_change=30
        #     cicrcshift_change=torch.randint(2*max_cicrcshift_change,(3,1)).view(-1).numpy()-max_cicrcshift_change
        #     img=np.roll(img,cicrcshift_change[0],axis=0)
        #     img=np.roll(img,cicrcshift_change[1],axis=1)
        #     img=np.roll(img,cicrcshift_change[2],axis=2)
            
            
            
            
        #     if torch.rand(1).numpy()[0]>0.2:
        #         shape=np.array(img.shape).astype(np.float)
        #         max_resize_change=0.3
    
        #         shape_new=np.round(shape*(1+torch.rand(3).numpy()*2*max_resize_change-max_resize_change))
        #         new_img=np.zeros(shape_new.astype(np.int),dtype=np.float32)
                
        #         in_1=np.zeros(3)
        #         in_2=np.zeros(3)
        #         out_1=np.zeros(3)
        #         out_2=np.zeros(3)
                
        #         for k in range(3):
        #             in_1[k]=np.floor(np.max([(shape[k]-shape_new[k])/2,0]))
        #             in_2[k]=np.min([in_1[k]+shape_new[k],shape[k]])
                    
        #             out_1[k]=np.floor(np.max([(shape_new[k]-shape[k])/2,0]))
        #             out_2[k]=np.min([out_1[k]+shape[k],shape_new[k]])
                    
        #         in_1=np.round(in_1).astype(np.int)
        #         in_2=np.round(in_2).astype(np.int)
                
        #         out_1=np.round(out_1).astype(np.int)
        #         out_2=np.round(out_2).astype(np.int)
                
        #         new_img[out_1[0]:out_2[0],out_1[1]:out_2[1],out_1[2]:out_2[2]]=img[in_1[0]:in_2[0],in_1[1]:in_2[1],in_1[2]:in_2[2]]
                
        #         factor=np.array(img.shape)/np.array(new_img.shape)
            
        #         img=zoom(new_img,factor,order=1,prefilter=False)

            
            

        
        

        
        img=np.expand_dims(img, axis=0).copy()
        img=torch.from_numpy(img)
        
        
        lbl=self.lbls[index]
        lbl2=np.zeros(self.rots_table.shape[0]).astype(np.float32)
        lbl2[lbl]=1
        
        
        lbl=torch.from_numpy(lbl2)
            
        return img,lbl
        
        
    
    
    
    
    
    

     