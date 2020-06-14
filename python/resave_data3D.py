
import numpy as np
import os
import glob
from scipy.io import loadmat



#  not working - some data starts with number ->   matfile are used as input



output_size=128


output_size_v=np.array([output_size,output_size,output_size])



folder = 'Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data_mat_' + str(output_size)





file_names=glob.glob(folder + os.sep + '**' + os.sep + '*.mat', recursive=True)


for file_num,file_name in enumerate(file_names):
    print(file_num)
    
    file_name_save=file_name
    file_name_save=file_name_save.replace('CT_rotation_data_mat_' + str(output_size),'CT_rotation_data_npy_'+ str(output_size))
    file_name_save=file_name_save.replace('.mat','.npy')

    head,tail = os.path.split(file_name_save) 
    
    
    
    try:
        os.makedirs(head)
    except:
        pass


    ct_scan=loadmat(file_name)
    
    ct_scan=ct_scan['data']
    
    np.save(file_name_save, ct_scan)
    
    









