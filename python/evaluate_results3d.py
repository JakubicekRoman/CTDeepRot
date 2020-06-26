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
from utils.get_2d_feature_imgs import get_2d_feature_imgs

import skimage.io as io

from scipy.io import savemat


# rotation=[90,180,270]
# rotation=[0,90,180]
# rotation=[90,90,0]
# rotation=[180,0,0]
# rotation=[270,0,180]
is3d=1



path=r'D:\vicar\tmp_romanovi_rotace\CT_rotation_data_x'

device = torch.device("cuda:0")




model=Small_resnet3D(input_size=1,output_size=24,lvl1_size=4)
model.load_state_dict(torch.load('../../models_python/Aug3D8_0.0001_train_0.9876055_valid_0.9995229_model.pt')) 

model=model.to(device)
model=model.eval()


data = pd.read_csv("utils/rot_dict_unique.csv") 
rots_table=data.loc[:,:].to_numpy()

xl_file = pd.ExcelFile(path + os.sep+'ListOfData.xlsx')
data = pd.read_excel(xl_file,header=None)

folders=data.loc[:,0].tolist()
names=data.loc[:,1].tolist()
file_names=[]
for folder,name in zip(folders,names):
    
    file_names.append((path + os.sep + folder.split('\\')[-1] + os.sep + 'x' + name + '.mhd'))


file_names=file_names[int(len(file_names)*0.8):]

file_names_all=[]
rots_gt=[]
difs=[]
rots_res=[]
psts=[]


for file_num,file_name in enumerate(file_names):
    print(file_num)
    
    try:
        orig_data = np.transpose(io.imread(file_name, plugin='simpleitk'),[1,2,0])
    except:
        print('fail!!!!!!!!!!!!!!!!!!!!!!!!')
        fsfdsffsdfs
        continue
    factor=np.array([128,128,128])/orig_data.shape
                
    orig_data=zoom(orig_data,factor,order=1)
    
    for a in [0,90,180,270]:
        for b in [0,90,180,270]:
            for c in [0,90,180,270]:
    
                rotation=[a,b,c]
                
                
            
                
                
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
                
                
                dif=np.sum(np.abs(fixed_data-orig_data))
                
                
                difs.append(dif)
                file_names_all.append(file_name)
                rots_gt.append(rotation)
                rots_res.append(pred_rot)
                psts.append(np.max(res))
                print(dif)
                
                
                # plt.figure(figsize=(15, 15))
    
                # plt.subplot(3,3,1)
                # plt.imshow(np.mean(orig_data,0))
                # plt.subplot(3,3,2)
                # plt.imshow(np.mean(orig_data,1))
                # plt.subplot(3,3,3)
                # plt.imshow(np.mean(orig_data,2))
                
                # plt.subplot(3,3,4)
                # plt.imshow(np.mean(rotated_data,0))
                # plt.subplot(3,3,5)
                # plt.imshow(np.mean(rotated_data,1))
                # plt.subplot(3,3,6)
                # plt.imshow(np.mean(rotated_data,2))
                
                # plt.subplot(3,3,7)
                # plt.imshow(np.mean(fixed_data,0))
                # plt.subplot(3,3,8)
                # plt.imshow(np.mean(fixed_data,1))
                # plt.subplot(3,3,9)
                # plt.imshow(np.mean(fixed_data,2))
                
                # plt.show()
                
              
    
              
mdic = {"difs": difs, "file_names_all": file_names_all,"rots_gt":rots_gt,"rots_res":rots_res,"psts":psts}

savemat("results_3d.mat", mdic)
    
acc=np.mean((np.array(difs)==0).astype(np.float))



print(acc)



