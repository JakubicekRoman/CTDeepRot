import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt

class myConv(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=None,do_batch=1,dov=0):
        super().__init__()
        
        pad=int((filter_size-1)/2)
        
        self.do_batch=do_batch
        self.dov=dov
        self.conv=nn.Conv3d(in_size, out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm3d(out_size,momentum=0.1)
        
        
        if self.dov>0:
            self.do=nn.Dropout(dov)
    
    def forward(self, inputs):
     
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)  
        
        outputs=F.relu(outputs)
        
        if self.dov>0:
            outputs = self.do(outputs)
        
        return outputs




class Small_resnet3D(nn.Module):
    
    
    def __init__(self, input_size,output_size,levels=5,lvl1_size=4):
        super().__init__()
        self.lvl1_size=lvl1_size
        self.levels=levels
        self.output_size=output_size
        self.input_size=input_size
        
        
        self.init_conv=myConv(input_size,lvl1_size)
        

        self.layers=nn.ModuleList()
        for lvl_num in range(levels):
            
            if lvl_num!=0:
                self.layers.append(myConv( int(lvl1_size*(lvl_num)), int(lvl1_size*(lvl_num+1))))
            
            self.layers.append(myConv( int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1))))
            
            self.layers.append(myConv( int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1))))
            
            self.layers.append(myConv( int(lvl1_size*(lvl_num+1)), int(lvl1_size*(lvl_num+1))))
            
        
        self.fc=nn.Linear(int(self.lvl1_size*self.levels*(128/2**self.levels)**3), output_size)
        
        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
        
        
    def forward(self, x):   
        
        
        y=self.init_conv(x)
        
        layer_num=-1
        for lvl_num in range(self.levels):
            
            
            if lvl_num!=0:
                layer_num=layer_num+1
                y=self.layers[layer_num](x)
            
            layer_num=layer_num+1
            x=self.layers[layer_num](y)
            layer_num=layer_num+1
            x=self.layers[layer_num](x)
            layer_num=layer_num+1
            x=self.layers[layer_num](x)
            
            x=x+y
            
            x=F.max_pool3d(x, 2, 2)
            
        
        shape=list(x.size())
        x=x.view(shape[0],-1)
        x=self.fc(x)
        
        
        return x
        


















