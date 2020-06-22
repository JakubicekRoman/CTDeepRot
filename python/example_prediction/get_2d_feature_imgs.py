import numpy as np
from mat2gray import mat2gray
from skimage.transform import resize 
from scipy.ndimage import zoom

def imresize(img,dim):
    tmp=resize(img,dim,order=3)
    return tmp

# def imresize(img,dim):
#     factor=(np.array(dim)/np.array(img.shape)).tolist()
#     tmp=zoom(img,factor,order=3)
#     return tmp


def get_2d_feature_imgs(data):
    
    data=data.astype(np.float32)
    
    size=224
    
    t1,t2 = 50,1800
    imMean=np.zeros((size,size,3),dtype=np.uint8)
    for dim in range(3):
        img = np.mean(data,dim)
        img=mat2gray(img,[t1,t2])
        img=imresize(img,[size,size])
        img=(img*255).astype(np.uint8)
        imMean[:,:,dim]=img
    
    
    t1,t2 = 100,3200
    imMax=np.zeros((size,size,3),dtype=np.uint8)
    for dim in range(3):
        img = np.max(data,dim)
        img=mat2gray(img,[t1,t2])
        img=imresize(img,[size,size])
        img=(img*255).astype(np.uint8)
        imMax[:,:,dim]=img

    
    
    t1,t2 = 0,1000
    imStd=np.zeros((size,size,3),dtype=np.uint8)
    for dim in range(3):
        img = np.std(data,dim,ddof=1)
        img=mat2gray(img,[t1,t2])
        img=imresize(img,[size,size])
        img=(img*255).astype(np.uint8)
        imStd[:,:,dim]=img
    
    return imMean,imMax,imStd