
import numpy as np

from scipy.ndimage import zoom
import os
import glob

from utils import load_itk



output_size=128


output_size_v=np.array([output_size,output_size,output_size])



for folder in ['D:\jakubicek\Rot_detection\data_3d\Data_raw_test','D:\jakubicek\Rot_detection\data_3d\Data_raw_train']:
    
    
    folder_save=folder
    folder_save=folder_save.replace('data_3d','data_3d_'+ str(output_size))
    
    try:
        os.makedirs(folder_save)
    except:
        pass
    
    
    file_names=glob.glob(folder + '/*.mhd', recursive=True)
    
    
    for file_num,file_name in enumerate(file_names):
        
        file_name_save=file_name
        file_name_save=file_name_save.replace('data_3d','data_3d_'+ str(output_size))
        file_name_save=file_name_save.replace('.mhd','.npy')
    
    
        ct_scan, origin, spacin = load_itk(file_name)
        
        factor=output_size_v/ct_scan.shape
        
        ct_scan_resized=zoom(ct_scan,factor)

        np.save(file_name_save, ct_scan_resized)
        
        









