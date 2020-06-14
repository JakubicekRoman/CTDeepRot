from config import Config

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



split='training'

xl_file = pd.ExcelFile(Config.data_path + os.sep +split+  os.sep+'labels_bin.xlsx')
data = pd.read_excel(xl_file,header=None)
file_names=data.loc[:,0].values.tolist()

folders = ['max_40','max_All','mean_20','mean_All','std_40','std_All']
Rs= ['R3','R4','R1','R2','R5','R6']

STDS = { i : 0 for i in folders  }
MEANS = { i : 0 for i in folders  }


for folder,R in zip(folders,Rs):
   
    means=[]
    stds=[]
    for k in range(3):

        meas_tmp=[]
        stds_tmp=[]
        for i,file_name in enumerate(file_names[::11]):
    
            tmp=imread(Config.data_path + os.sep +split +  os.sep +  folder +  os.sep + file_name  +'_' + R + '_Ch' + str(k+1)  + '.png')
            tmp=tmp.astype(np.float32)/255
            
            tmp=resize(tmp,[224,224])
            
            meas_tmp.append(np.mean(tmp))
            stds_tmp.append(np.std(tmp))
            
        means.append(np.mean(meas_tmp))
        stds.append(np.mean(stds_tmp))
            
    MEANS[folder]=means
    STDS[folder]=stds     
            
            
            
            
            
            
            
            