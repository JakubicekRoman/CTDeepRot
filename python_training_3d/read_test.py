import time
import numpy as np
from scipy.io import loadmat


start = time.time()

for k in range(100):
    # a=loadmat("D:\jakubicek\Rot_detection\data_3d_128_mat\Data_raw_test\VerSe20_0001.mat")
    img=np.load("D:\jakubicek\Rot_detection\data_3d_128\Data_raw_test\VerSe20_0001.npy")
    
end = time.time()

print(end - start)