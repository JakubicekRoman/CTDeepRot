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


class DataLoader(data.Dataset):
    
    def __init__(self, split,path_to_data):
    
        self.split=split
        self.path=path_to_data
        
        
        
        xl_file = pd.ExcelFile(self.path + os.sep +self.split+  os.sep+'labels_bin.xlsx')

        data = pd.read_excel(xl_file,header=None)
        
        
        file_names_=data[0].values.tolist()
        
        labels=data.loc[:,1:7].to_numpy()
        
        
        
        
        
    
    
    
    def __len__(self):
        pass
    
    
    
    def __getitem__(self, index):
        pass

     