import torch
import numpy as np
import matplotlib.pyplot as plt


def wce(res,lbls,w_positive_tensor,w_negative_tensor):
    ## weighted crossetropy - weigths are for positive and negative 
    res_c = torch.clamp(res,min=1e-6,max=1-1e-6)
            
    p1=lbls*torch.log(res_c)*w_positive_tensor
    p2=(1-lbls)*torch.log(1-res_c)*w_negative_tensor
    
    return -torch.mean(p1+p2)



def Log():
    def __init__(self):
        self.model_names=[]
            
        self.trainig_loss_log=[]
        self.test_loss_log=[]
        
        self.trainig_loss_log_tmp=[]
        self.test_loss_log_tmp=[]
        
        
        self.trainig_acc_log=[]
        self.test_acc_log=[]
        
        self.trainig_loss_log_tmp=[]
        self.test_acc_log_tmp=[]
        
        
        
    def append_train(self,loss,acc):
        self.trainig_loss_log_tmp.append(loss.detach().cpu().numpy())
        self.trainig_acc_log_tmp.append(acc.detach().cpu().numpy())
        
        
    def append_test(self,loss,acc):
        self.test_loss_log_tmp.append(loss.detach().cpu().numpy())
        self.test_acc_log_tmp.append(acc.detach().cpu().numpy())
        
        
    def save_and_reset(self):
        
        self.trainig_loss_log.append(np.mean(self.trainig_loss_log_tmp))
        self.test_loss_log.append(np.mean(self.test_loss_log_tmp))
        
        self.trainig_loss_log_tmp=[]
        self.test_loss_log_tmp=[]
        
        
        self.trainig_acc_log.append(np.mean(self.trainig_acc_log_tmp))
        self.test_acc_log.append(np.mean(self.test_acc_log_tmp))
        
        self.trainig_acc_log_tmp=[]
        self.test_acc_log_tmp=[]
        
        
    def plot(self):
        
        plt.plot( self.trainig_loss_log, label = 'training')
        plt.plot(self.test_loss_log, label = 'test')
        plt.tilte('loss')
        plt.show()
        
        plt.plot( self.trainig_acc_log, label = 'training')
        plt.plot(self.test_acc_log, label = 'test')
        plt.tilte('acc')
        plt.show()
        
        
        