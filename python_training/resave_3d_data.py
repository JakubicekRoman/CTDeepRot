import SimpleITK as sitk
import numpy as np

from scipy.ndimage import zoom
import os
import glob


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


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
    
    
        ct_scan, origin, spacin = load_itk("D:\jakubicek\Rot_detection\data_3d\Data_raw_test\VerSe20_0001.mhd")
        
        factor=output_size_v/ct_scan.shape
        
        ct_scan_resized=zoom(ct_scan,factor)

        np.save(file_name_save, ct_scan_resized)
        
        









