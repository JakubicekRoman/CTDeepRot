clear all;close all force;clc;

file_name='Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data\VerSe2020_train\VerSe20_0090.mhd';

orig_data=single(load_raw(file_name));

orig_data(orig_data<-1024)=nan;
if nanmin(orig_data(:))<0
    orig_data=orig_data+1024;
end
orig_data(isnan(orig_data))=0;


rotated_data=rotate_3d(orig_data,[90,180,270]);


% [fixed_data,angles] = predict2d(rotated_data);
[fixed_data,angles] = predict3d(rotated_data);


figure()
subplot(1,3,1)
imshow(squeeze(mean(orig_data,1)),[])
title('coronal')
xlabel('cranial<------> caudal')
subplot(1,3,2)
imshow(squeeze(mean(orig_data,2)),[])
title('sagital')
xlabel('cranial <------> caudal')
subplot(1,3,3)
imshow(squeeze(mean(orig_data,3)),[])
title('axial')
xlabel('left/rigth <------> rigth/left')
sgtitle('Ground Truth')


figure()
subplot(1,3,1)
imshow(squeeze(mean(rotated_data,1)),[])
title('coronal')
xlabel('cranial<------> caudal')
subplot(1,3,2)
imshow(squeeze(mean(rotated_data,2)),[])
title('sagital')
xlabel('cranial <------> caudal')
subplot(1,3,3)
imshow(squeeze(mean(rotated_data,3)),[])
title('axial')
xlabel('left/rigth <------> rigth/left')
sgtitle('Input')



figure()
subplot(1,3,1)
imshow(squeeze(mean(fixed_data,1)),[])
title('coronal')
xlabel('cranial<------> caudal')
subplot(1,3,2)
imshow(squeeze(mean(fixed_data,2)),[])
title('sagital')
xlabel('cranial <------> caudal')
subplot(1,3,3)
imshow(squeeze(mean(fixed_data,3)),[])
title('axial')
xlabel('left/rigth <------> rigth/left')
sgtitle('Results')







