from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import DataLoader

from utils import wce


from torchvision import models

from config import Config



# if __name__ == '__main__':


device = torch.device("cuda:0")
    
        
w_positive=np.zeros(7)
w_negative=np.zeros(7)
w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).to(device)
w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).to(device)



loader = DataLoader(split='training',path_to_data=Config.data_path)
trainloader= data.DataLoader(loader, batch_size=Config.train_batch_size, num_workers=0, shuffle=True,drop_last=True)

loader = DataLoader(split='testing',path_to_data=Config.data_path)
testLoader= data.DataLoader(loader, batch_size=Config.train_batch_size, num_workers=0, shuffle=False,drop_last=False)



model = models.resnet50(pretrained=False,num_classes=7)
model.conv1 = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False)
model=model.to(device)




for epoch_num in range(Config.max_epochs):
    
    for it, (batch,lbls) in enumerate(trainloader): ### you can iterate over dataset (one epoch)
        
        
        batch=batch.to(device)
        lbls=lbls.to(device)
        
        res=model(batch)
        res=torch.sigmoid(res)
        
        loss = wce(res,lbls,w_positive_tensor,w_negative_tensor)
        
        
        asdffdsfs

        
        
        
        
        
    for it, (batch,lbls) in enumerate(testLoader): ### you can iterate over dataset (one epoch)
       
       res=model(batch)
       res=torch.sigmoid(res)
        
        
        

    
    
    

        
        
    
    
    
    















