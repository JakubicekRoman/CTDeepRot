import torch 
from rotate_fcns import rotate_3d,rotate_3d_inverse
import numpy as np
import pandas as pd 
from small_resnet3D import Small_resnet3D
from scipy.ndimage import zoom

def predict_3d(data,device="cuda:0"):
    
    img=data.copy()
    
    device = torch.device(device)

    model=Small_resnet3D(input_size=1,output_size=24,lvl1_size=4)
    model.load_state_dict(torch.load('models/net3d.pt')) 
    
    
    model=model.to(device)
    model=model.eval()
    
    
    
    factor=np.array([128,128,128])/img.shape
                
    img=zoom(img,factor,order=1)
            
            
            

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
    
    
    tmp = pd.read_csv("rot_dict_unique.csv") 
    rots_table=tmp.loc[:,:].to_numpy()
    
    
    pred_rot=rots_table[pred,:3]
    angles=pred_rot.tolist()[0]
    fixed_data=data.copy()
    fixed_data=rotate_3d_inverse(fixed_data,angles)
    
    return fixed_data,angles