
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

import pandas as pd 

from skimage.transform import resize


path= '../../../CT_rotation_data_2D'


xl_file = pd.ExcelFile(path + os.sep+'ListOfData.xlsx')
data = pd.read_excel(xl_file,header=None)

folders=data.loc[:,0].tolist()
names=data.loc[:,1].tolist()
file_names=[]
for folder,name in zip(folders,names):
    
    file_names.append((path + os.sep + folder.split('\\')[-1] + os.sep + name).replace('.mhd',''))


file_names=file_names[:int(len(file_names)*0.8)]

    
folders=['mean','max','std']

    
STDS = { i : 0 for i in folders  }
MEANS = { i : 0 for i in folders  }



for folder in folders:
   
    means=[]
    stds=[]
    for k in range(3):

        meas_tmp=[]
        stds_tmp=[]
        for i,file_name in enumerate(file_names):
            print(i)
    
            tmp=imread(file_name + '_' + folder + '_'+ str(k+1)  +'.png' )
            tmp=tmp.astype(np.float32)/255
            
            
            meas_tmp.append(np.mean(tmp))
            stds_tmp.append(np.std(tmp))
            
        means.append(np.mean(meas_tmp))
        stds.append(np.mean(stds_tmp))
            
    MEANS[folder]=means
    STDS[folder]=stds     
            
            
            
            
            
            
            
            