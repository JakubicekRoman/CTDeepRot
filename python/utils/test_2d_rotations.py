import numpy as np
from rotate_fcns import rotate_2d,rotate_3d,rotate_3d_inverse,flip_2d
import matplotlib.pyplot as plt


data=np.load(r"D:\vicar\tmp_romanovi_rotace\CT_rotation_data_npy_128\VerSe2019_test\VerSe20_0001.npy")



rotace=[90,270,90]


# data_rot=rotate_3d(data,rotace)



# tmp=(np.squeeze(np.mean(data_rot,0)),np.squeeze(np.mean(data_rot,1)),np.squeeze(np.mean(data_rot,2)))
# data_2d_rot1=np.stack(tmp,2)


# tmp=(np.squeeze(np.mean(data,0)),np.squeeze(np.mean(data,1)),np.squeeze(np.mean(data,2)))
# data_2d=np.stack(tmp,2)
# data_2d_rot2=rotate_2d(data_2d,rotace)




# tmp=(np.squeeze(np.mean(data,0)),np.squeeze(np.mean(data,1)),np.squeeze(np.mean(data,2)))
# data_2d_rot1=np.stack(tmp,2)


# data_rot=rotate_3d_inverse(data_rot,rotace)
# tmp=(np.squeeze(np.mean(data_rot,0)),np.squeeze(np.mean(data_rot,1)),np.squeeze(np.mean(data_rot,2)))
# data_2d_rot2=np.stack(tmp,2)




tmp=(np.squeeze(np.mean(data,0)),np.squeeze(np.mean(data,1)),np.squeeze(np.mean(data,2)))
data_2d_rot1=np.stack(tmp,2)

data_2d_rot2=flip_2d(data_2d_rot1)



plt.imshow(data_2d_rot1[:,:,0])
plt.show()
plt.imshow(data_2d_rot1[:,:,1])
plt.show()
plt.imshow(data_2d_rot1[:,:,2])
plt.show()


plt.imshow(data_2d_rot2[:,:,0])
plt.show()
plt.imshow(data_2d_rot2[:,:,1])
plt.show()
plt.imshow(data_2d_rot2[:,:,2])
plt.show()