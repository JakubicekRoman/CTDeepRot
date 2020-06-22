import torch 
from rotate_fcns import rotate_3d,rotate_3d_inverse
import numpy as np
from torchvision import models
import torch.nn as nn
from get_2d_feature_imgs import get_2d_feature_imgs
import pandas as pd 

def predict_2d(data,device="cuda:0"):
    
    
    device = torch.device(device)

    model = models.resnet18(pretrained=0)
    model.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 24)
    model.load_state_dict(torch.load('models/net2d.pt')) 
    
    
    model=model.to(device)
    model=model.eval()
    
    
    
    img=data
    
    imMean,imMax,imStd=get_2d_feature_imgs(img)
                
    img_list=[imMean[:,:,0],imMean[:,:,1],imMean[:,:,2],imMax[:,:,0],imMax[:,:,1],imMax[:,:,2],imStd[:,:,0],imStd[:,:,1],imStd[:,:,2]]
    
    
    folders=['mean','max','std']
    
    ind=-1
    for folder in folders:
        for k in range(3):
            ind=ind+1
            tmp=img_list[ind]
            tmp=tmp.astype(np.float32)/255-0.5
            img_list[ind]=tmp
            
            
            

    img=np.stack(img_list,axis=2)
    
    img=np.transpose(img,(2,0,1))
    img=np.expand_dims(img, axis=0).copy()
    
    img=torch.from_numpy(img)
    img=img.to(device)
    
    res=model(img)
    res=torch.softmax(res,1)
    res=res.detach().cpu().numpy()
    pred=np.argmax(res,1)
    
    
    tmp = pd.read_csv("rot_dict_unique.csv") 
    rots_table=tmp.loc[:,:].to_numpy()
    
    
    pred_rot=rots_table[pred,:3]
    angles=pred_rot.tolist()[0]
    fixed_data=data.copy()
    fixed_data=rotate_3d_inverse(fixed_data,angles)
    
    return fixed_data,angles