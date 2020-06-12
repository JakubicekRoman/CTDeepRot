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
from utils import angle2vec



class DataLoader(data.Dataset):
    
    def __init__(self, split,path_to_data):
    
        self.split=split
        self.path=path_to_data
        
        
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
            file_names=file_names[int(len(file_names)*0.8):]
            
        
        self.file_names=[]
        self.vec=[]
        self.flip=[]
        for file in file_names:
            for flip in [0,1]:
                unique_rots=[]
                for a in [0,90,180,270]:
                    for b in [0,90,180,270]:
                        for c in [0,90,180,270] :
                            
                            rot=np.array([a,b,c])
                            rot_vec =angle2vec(rot)
                            
                            new=np.sum([all(unique_rot==rot_vec) for unique_rot in unique_rots])==0
                            
                            if new:
                                unique_rots.append(rot_vec)
                                self.file_names.append(file)
                                self.vec.append(rot)
                                self.flip.append(flip)
                                

                            
                            
    
    def __len__(self):
        return len(self.file_names)
    
    
    
    def __getitem__(self, index):
        
        file_name=self.file_names[index]
        
        MEAN=614.2868
        STD=614.2868
        img=np.load(file_name)
        img=img.astype(np.float32)
        img=(img-MEAN)/STD
        
            



        a,b,c=self.vec[index]
        flip=self.flip[index]
        a=np.array([a])
        b=np.array([b])
        c=np.array([c])
        flip=np.array([flip])
            
        # if flip:
        #     img=img[:,:,::-1]
            
            
        img=np.rot90(img,int(a/90),axes=(0,1))
        img=np.rot90(img,int(b/90),axes=(0,2))
        img=np.rot90(img,int(c/90),axes=(1,2))
        
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
        
        
        lbls_angle=np.concatenate((a,b,c))
        lbl2=angle2vec(lbls_angle)
        
        
        lbl=torch.from_numpy(lbl2)
            
        return img,lbl
        
        
    
    
    
    
    
    

     