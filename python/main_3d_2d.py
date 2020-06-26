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
from dataloader3D_2D import DataLoader3D_2D

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
    

    loader = DataLoader3D_2D(split='training',path_to_data=Config.data_path)
    trainloader= data.DataLoader(loader, batch_size=Config.train_batch_size, num_workers=Config.train_num_workers, shuffle=True,drop_last=True)
    
    loader = DataLoader3D_2D(split='testing',path_to_data=Config.data_path)
    testLoader= data.DataLoader(loader, batch_size=Config.test_batch_size, num_workers=Config.test_num_workers, shuffle=False,drop_last=False)
    
    (batch,s,lbls)=next(iter(trainloader))
    predicted_size=list(lbls.size())[1]
    input_size=list(batch.size())[1]
    
    
    
    class prasacky_model(nn.Module):

        def __init__(self):
            super().__init__()
            self.model3d=Small_resnet3D(input_size=1,output_size=100,lvl1_size=3)
            self.model2d = models.resnet18(pretrained=Config.pretrained)
            self.model2d.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.model2d.fc.in_features
            self.model2d.fc = torch.nn.Linear(num_ftrs, 100)
            
            self.fc1=nn.Linear(200, 300)
            
            self.fc2=nn.Linear(300, 300)
            
            self.fc3=nn.Linear(300, predicted_size)
            
            
            
        def forward(self, x3d,x2d):
            a=self.model3d(x3d)
            b=self.model2d(x2d)
            
            y=self.fc1(torch.cat((a,b),1))
            y=F.relu(y)
            y=self.fc2(y)
            y=F.relu(y)
            y=self.fc3(y)
            
            return y



    model=prasacky_model()

    
    model=model.to(device)
    
    
    
    optimizer = optim.Adam(model.parameters(),lr=Config. init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.StepLR(optimizer, Config.step_size, gamma=Config.gamma, last_epoch=-1)
    
    log = Log()
    for epoch_num in range(Config.max_epochs):
        
        model.train()
        N=len(trainloader)
        for it, (batch3d,batch2d,lbls) in enumerate(trainloader):
            if it%50==0:
                print(str(it) + '/' + str(N))
            
            
            batch3d=batch3d.to(device)
            batch2d=batch2d.to(device)
            lbls=lbls.to(device)
            
            res=model(batch3d,batch2d)
            
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
        for it, (batch3d,batch2d,lbls) in enumerate(testLoader): 
            if it%50==0:
                print(str(it) + '/' + str(N))
           
            batch3d=batch3d.to(device)
            batch2d=batch2d.to(device)
            lbls=lbls.to(device)
            
            res=model(batch3d,batch2d)
            
            res=torch.softmax(res,1)
            loss = ce(res,lbls)
            
            
            loss=loss.detach().cpu().numpy()
            res=res.detach().cpu().numpy()
            lbls=lbls.detach().cpu().numpy()

            acc=np.mean((np.argmax(res,1)==np.argmax(lbls,1)).astype(np.float32))
            
    
            log.append_test(loss,acc)
           
           
           
            
        log.save_and_reset()
         
        
        
        info= str(epoch_num) + '_' + str(get_lr(optimizer)) + '_train_'  + str(log.trainig_acc_log[-1]) + '_valid_' + str(log.test_acc_log[-1]) 
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
        
        
        
    
    
    

        
        
    
    
    
    















