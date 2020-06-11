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


xl_file = pd.ExcelFile(Config.data_path + os.sep+'ListOfData.xlsx')
data = pd.read_excel(xl_file,header=None)

folders=data.loc[:,0].tolist()
names=data.loc[:,1].tolist()
file_names=[]
for folder,name in zip(folders,names):
    
    file_names.append((Config.data_path + os.sep + folder.split('\\')[-1] + os.sep + name).replace('.mhd','.npy'))



meas_tmp=[]
stds_tmp=[]

for i,file_name in enumerate(file_names[::3]):
    print(i)

    tmp=np.load(file_name)
    tmp=tmp.astype(np.float32)
    
    meas_tmp.append(np.mean(tmp))
    stds_tmp.append(np.std(tmp))
    

    
MEAN=np.mean(meas_tmp)
STD=np.mean(stds_tmp)   
            
            
            
            
            
            
            
            