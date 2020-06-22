import numpy as np

def mat2gray(data,min_max=None):
    
    if min_max==None:
        minv=np.min(data)
        maxv=np.max(data)
    else:
        minv=min_max[0]
        maxv=min_max[1]
        
    data[data<minv]=minv
    data[data>maxv]=maxv
    
    data=(data-minv)/(maxv-minv)
    
    return data
    
    
    
    
