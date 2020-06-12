clc;clear all;close all;


load("D:\vicar\tmp_romanovi_rotace\CT_rotation_data_mat_128\VerSe2019_test\VerSe20_0001.mat")


rotace=[90,180,270];


data_rot=rotate_3d(data,rotace);


data_2d_rot1=cat(3,squeeze(mean(data_rot,1)),squeeze(mean(data_rot,2)),squeeze(mean(data_rot,3)));

data_2d=cat(3,squeeze(mean(data,1)),squeeze(mean(data,2)),squeeze(mean(data,3)));

data_2d_rot2=rotate_2d(data_2d,rotace);


figure();
subplot(2,3,1)
imshow(data_2d_rot1(:,:,1),[])
subplot(2,3,2)
imshow(data_2d_rot1(:,:,2),[])
subplot(2,3,3)
imshow(data_2d_rot1(:,:,3),[])



subplot(2,3,4)
imshow(data_2d_rot2(:,:,1),[])
subplot(2,3,5)
imshow(data_2d_rot2(:,:,2),[])
subplot(2,3,6)
imshow(data_2d_rot2(:,:,3),[])

