import skimage.io as io
from rotate_fcns import rotate_3d,rotate_3d_inverse
import numpy as np
from predict_2d import predict_2d
from predict_3d import predict_3d
import matplotlib.pyplot as plt


file_name="Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data\VerSe2020_train\VerSe20_0090.mhd"


orig_data = np.transpose(io.imread(file_name, plugin='simpleitk'),[1,2,0])


rotated_data=orig_data.copy()
rotated_data=rotate_3d(rotated_data,[90,180,270])


fixed_data,angles = predict_2d(rotated_data)
# fixed_data,angles = predict_3d(rotated_data)


plt.figure(figsize=(15, 7))
plt.suptitle('Ground Truth')
plt.subplot(1,3,1)
plt.imshow(np.mean(orig_data,0))
plt.title('coronal')
plt.xlabel('cranial<------> caudal')
plt.subplot(1,3,2)
plt.imshow(np.mean(orig_data,1))
plt.title('sagital')
plt.xlabel('cranial <------> caudal')
plt.subplot(1,3,3)
plt.imshow(np.mean(orig_data,2))
plt.title('axial')
plt.xlabel('left/rigth <------> rigth/left')
plt.show()

plt.figure(figsize=(15, 7))
plt.suptitle('Input')
plt.subplot(1,3,1)
plt.imshow(np.mean(orig_data,0))
plt.title('coronal')
plt.xlabel('cranial<------> caudal')
plt.subplot(1,3,2)
plt.imshow(np.mean(orig_data,1))
plt.title('sagital')
plt.xlabel('cranial <------> caudal')
plt.subplot(1,3,3)
plt.imshow(np.mean(orig_data,2))
plt.title('axial')
plt.xlabel('left/rigth <------> rigth/left')
plt.show()

plt.figure(figsize=(15, 7))
plt.suptitle('Results')
plt.subplot(1,3,1)
plt.imshow(np.mean(orig_data,0))
plt.title('coronal')
plt.xlabel('cranial<------> caudal')
plt.subplot(1,3,2)
plt.imshow(np.mean(orig_data,1))
plt.title('sagital')
plt.xlabel('cranial <------> caudal')
plt.subplot(1,3,3)
plt.imshow(np.mean(orig_data,2))
plt.title('axial')
plt.xlabel('left/rigth <------> rigth/left')
plt.show()






