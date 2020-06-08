from utils import load_itk
import numpy as np

from scipy.ndimage import zoom
import os
import glob

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F





MEAN=900.3071
STD=318.11615
 

output_size=128
output_size_v=np.array([output_size,output_size,output_size])





folder='D:\jakubicek\Rot_detection\data_3d\Data_raw_test'




file_names=glob.glob(folder + '/*.mhd', recursive=True)




for file_num, file_name in enumerate(file_names):
    
    
    




