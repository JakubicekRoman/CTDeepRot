from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader3D import DataLoader3D
from dataloader2D import DataLoader2D

from small_resnet3D import Small_resnet3D

from utils.utils import ce


from torchvision import models

from config import Config

from utils.utils import Log

from utils.utils  import get_lr

import pickle




if __name__ == '__main__':

    
    device = torch.device("cuda:0")
        
            
    try:
        os.mkdir(Config.tmp_save_dir)
    except:
        pass
    
    if Config.is3d:
        loader = DataLoader3D(split='training',path_to_data=Config.data_path)
    else:
        loader = DataLoader2D(split='training',path_to_data=Config.data_path)
    trainloader= data.DataLoader(loader, batch_size=Config.train_batch_size, num_workers=Config.train_num_workers, shuffle=True,drop_last=True)
    
    if Config.is3d:
        loader = DataLoader3D(split='testing',path_to_data=Config.data_path)
    else:
        loader = DataLoader2D(split='testing',path_to_data=Config.data_path)
    testLoader= data.DataLoader(loader, batch_size=Config.test_batch_size, num_workers=Config.test_num_workers, shuffle=False,drop_last=False)
    
    (batch,lbls)=next(iter(trainloader))
    predicted_size=list(lbls.size())[1]
    input_size=list(batch.size())[1]
    
    
    
    if Config.is3d:
        model=Small_resnet3D(input_size=1,output_size=predicted_size,lvl1_size=Config.lvl1_size)
        # model.load_state_dict(torch.load('3dmodel.pt')) 
    else:
        model = models.resnet50(pretrained=Config.pretrained)
        model.conv1 = nn.Conv2d(input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, predicted_size)
    model=model.to(device)
    
    optimizer = optim.Adam(model.parameters(),lr=Config. init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.StepLR(optimizer, Config.step_size, gamma=Config.gamma, last_epoch=-1)
    
    log = Log()
    for epoch_num in range(Config.max_epochs):
        
        model.train()
        N=len(trainloader)
        for it, (batch,lbls) in enumerate(trainloader):
            if it%1==0:
                print(str(it) + '/' + str(N))
            
            
            batch=batch.to(device)
            lbls=lbls.to(device)
            
            res=model(batch)
            
            res=torch.softmax(res,1)
            loss = ce(res,lbls)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            acc=np.mean((np.argmax(res,1)==np.argmax(lbls,1)).astype(np.float32))
            
            log.append_train(loss,acc)
    

            
            
        model.eval()    
        N=len(testLoader)
        for it, (batch,lbls) in enumerate(testLoader): 
            if it%1==0:
                print(str(it) + '/' + str(N))
           
            batch=batch.to(device)
            lbls=lbls.to(device)
            
            res=model(batch)
            
            res=torch.softmax(res,1)
            loss = ce(res,lbls)
            
            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            acc=np.mean((np.argmax(res,1)==np.argmax(lbls,1)).astype(np.float32))
            
    
            log.append_test(loss,acc)
           
           
           
            
        log.save_and_reset()
         
        
        
        info= str(epoch_num) + '_' + str(get_lr(optimizer)) + '_train_'  + str(log.test_acc_log[-1]) + '_valid_' + str(log.trainig_acc_log[-1]) 
        print(info)
        log.plot()
        
        
        scheduler.step()

        tmp_file_name= Config.tmp_save_dir + os.sep +Config.model_name + info
        torch.save(model.state_dict(),tmp_file_name +  '_model.pt')
        log.save_plot(tmp_file_name +  '_plot.png')
        
        with open(tmp_file_name +  '_log.pkl', 'wb') as f:
            pickle.dump(log, f)
            
        with open(tmp_file_name +  '_config.pkl', 'wb') as f:
            pickle.dump(Config(), f)
        
        
        
    
    
    

        
        
    
    
    
    















