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

from utils import Log



if __name__ == '__main__':

    
    device = torch.device("cuda:0")
        
            
    w_positive=np.ones(7)
    w_negative=np.ones(7)
    w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).to(device)
    w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).to(device)
    
    
    
    loader = DataLoader(split='training',path_to_data=Config.data_path)
    trainloader= data.DataLoader(loader, batch_size=Config.train_batch_size, num_workers=Config.train_num_workers, shuffle=True,drop_last=True)
    
    loader = DataLoader(split='testing',path_to_data=Config.data_path)
    testLoader= data.DataLoader(loader, batch_size=Config.test_batch_size, num_workers=Config.test_num_workers, shuffle=False,drop_last=False)
    
    
    
    model = models.resnet50(pretrained=False,num_classes=7)
    model.conv1 = nn.Conv2d(18, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model=model.to(device)
    
    optimizer = optim.Adam(model.parameters(),lr=Config. init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.StepLR(optimizer, Config.step_size, gamma=Config.gamma, last_epoch=-1)
    
    
    
    log = Log()
    for epoch_num in range(Config.max_epochs):
        
        for it, (batch,lbls) in enumerate(trainloader): ### you can iterate over dataset (one epoch)
            print(it)
            
            
            batch=batch.to(device)
            lbls=lbls.to(device)
            
            res=model(batch)
            res=torch.sigmoid(res)
            
            loss = wce(res,lbls,w_positive_tensor,w_negative_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            acc=torch.mean((torch.sum(lbls==(res>0.5),1)==7).type(torch.float32))
            
            
            log.append_train(loss,acc)
    
            
            
            
            
            
        for it, (batch,lbls) in enumerate(testLoader): ### you can iterate over dataset (one epoch)
           
            batch=batch.to(device)
            lbls=lbls.to(device)
            
            res=model(batch)
            res=torch.sigmoid(res)
            
            loss = wce(res,lbls,w_positive_tensor,w_negative_tensor)
            
            acc=torch.mean((torch.sum(lbls==(res>0.5),1)==7).type(torch.float32))
            
            log.append_test(loss,acc)
           
           
           
            
        log.save_and_reset()
         
        log.plot()
        
        scheduler.step()




    
    
    

        
        
    
    
    
    















