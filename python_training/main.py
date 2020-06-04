from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from torch import optim

from dataloader import DataLoader

from sit import Sit

from torchvision import models

from config import Config


if __name__ == '__main__':
    
    
    loader = DataLoader(split='training',Config.data_path)
    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=0, shuffle=True,drop_last=True)
    
    # loader = DataLoader(split='testing')
    # validloader= data.DataLoader(loader, batch_size=batch_valid, num_workers=0, shuffle=False,drop_last=False)




















