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





file_names=glob.glob(Config.data_path + os.sep + 'Data_raw_train' + os.sep + '*.npy', recursive=True)


meas_tmp=[]
stds_tmp=[]

for i,file_name in enumerate(file_names[::3]):

    tmp=np.load(file_name)
    tmp=tmp.astype(np.float32)
    
    meas_tmp.append(np.mean(tmp))
    stds_tmp.append(np.std(tmp))
    

    
MEAN=np.mean(meas_tmp)
STD=np.mean(stds_tmp)   
            
            
            
            
            
            
            
            