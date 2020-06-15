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

from utils.rotate_fcns import rotate_2d,rotate_3d,rotate_3d_inverse

from torchvision import models

from small_resnet3D import Small_resnet3D



# rotation=[90,180,270]
# rotation=[0,90,180]
# rotation=[90,90,0]
# rotation=[180,0,0]
rotation=[270,0,180]
is3d=1


if is3d:
    path='../../CT_rotation_data_npy_128'
else:
    path='../../CT_rotation_data_2D'

device = torch.device("cuda:0")




if is3d:
    model=Small_resnet3D(input_size=1,output_size=24,lvl1_size=4)
    model.load_state_dict(torch.load('../../models_python/Aug3D8_0.0001_train_0.9876055_valid_0.9995229_model.pt')) 
else:
    model = models.resnet50(pretrained=0)
    model.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 24)
    model.load_state_dict(torch.load('../../models_python/NoAug2D10_1e-05_train_0.9766281_valid_0.99909806_model.pt')) 

model=model.to(device)



data = pd.read_csv("utils/rot_dict_unique.csv") 
rots_table=data.loc[:,:].to_numpy()

xl_file = pd.ExcelFile(path + os.sep+'ListOfData.xlsx')
data = pd.read_excel(xl_file,header=None)

folders=data.loc[:,0].tolist()
names=data.loc[:,1].tolist()
file_names=[]
for folder,name in zip(folders,names):
    
    file_names.append((path + os.sep + folder.split('\\')[-1] + os.sep + name).replace('.mhd','.npy'))


file_names=file_names[-19:-13]   


for file_num,file_name in enumerate(file_names):
    
    orig_data=np.load(file_name)
    
    rotated_data=orig_data.copy()
    rotated_data=rotate_3d(rotated_data,rotation)
    
    
    img=rotated_data.copy()
    img=img.astype(np.float32)
    MEAN=614.2868
    STD=614.2868
    img=(img-MEAN)/STD
        
    
    img=np.expand_dims(img, axis=0)
    img=np.expand_dims(img, axis=0).copy()
    img=torch.from_numpy(img)
    img=img.to(device)
    
    res=model(img)
    res=torch.softmax(res,1)
    res=res.detach().cpu().numpy()
    pred=np.argmax(res,1)
    
    
    pred_rot=rots_table[pred,:3]
    fixed_data=rotated_data.copy()
    fixed_data=rotate_3d_inverse(fixed_data,pred_rot.tolist()[0])
    
    
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3,3,1)
    plt.imshow(np.mean(orig_data,0))
    plt.subplot(3,3,2)
    plt.imshow(np.mean(orig_data,1))
    plt.subplot(3,3,3)
    plt.imshow(np.mean(orig_data,2))
    
    plt.subplot(3,3,4)
    plt.imshow(np.mean(rotated_data,0))
    plt.subplot(3,3,5)
    plt.imshow(np.mean(rotated_data,1))
    plt.subplot(3,3,6)
    plt.imshow(np.mean(rotated_data,2))
    
    plt.subplot(3,3,7)
    plt.imshow(np.mean(fixed_data,0))
    plt.subplot(3,3,8)
    plt.imshow(np.mean(fixed_data,1))
    plt.subplot(3,3,9)
    plt.imshow(np.mean(fixed_data,2))
    
    plt.show()
    
    
    
    





