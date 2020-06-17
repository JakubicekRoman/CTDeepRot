import numpy as np
from utils.mat2gray import mat2gray
from skimage.transform import resize as imresize

def get_2d_feature_imgs(data):
    
    data=data.astype(np.float32)
    
    size=224
    
    t1,t2 = 50,1800;
    imMean=np.zeros((size,size,3))
    for dim in range(3):
        img = np.mean(data,dim)
        img=mat2gray(img,[t1,t2])
        img=imresize(img,[size,size],order=3)
        img=(img*255).astype(np.uint8)
        imMean[:,:,dim]=img
    
    
    t1,t2 = 100,3200
    imMax=np.zeros((size,size,3))
    for dim in range(3):
        img = np.max(data,dim)
        img=mat2gray(img,[t1,t2])
        img=imresize(img,[size,size],order=3)
        img=(img*255).astype(np.uint8)
        imMax[:,:,dim]=img

    
    
    t1,t2 = 0,1000
    imStd=np.zeros((size,size,3))
    for dim in range(3):
        img = np.std(data,dim,ddof=1)
        img=mat2gray(img,[t1,t2])
        img=imresize(img,[size,size],order=3)
        img=(img*255).astype(np.uint8)
        imStd[:,:,dim]=img
    
    return imMean,imMax,imStd