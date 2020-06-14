%% finalni program pro rotaci

clear all
close all
clc


load('Trained_nets\Net_3_regr_1.mat','net')

path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\DIR\raw\' 'DIR_Data_3.mhd' ];
% path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\Dataset_3_CT_Glocker_242_pat_vyber\Data_raw\' '4557469.mhd' ];
% path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\Dataset_3_CT_Glocker_242_pat_vyber\Data_raw\' '4574668.mhd' ];
% path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\vc prone and supine 3 cases\raw\' 'colon_sups_3.mhd' ];

im = load_raw(path_raw);

imOrig = create_features(im);

rot = [90,90,180];
% im = imrotate3(im,rot(1),[1,0,0],'nearest');
im = rot90_3D(im, 1, rot(1)/90);
% im = imrotate3(im,rot(2),[0,1,0],'nearest'); % rotY - rotace podle X osy
im = rot90_3D(im, 2, rot(2)/90);
% im = imrotate3(im,rot(3),[0,0,1],'nearest'); % rotZ - rotace AXIAL    
im = rot90_3D(im, 3, rot(3)/90);

codingAngle(rot)


%% nalezeni uhlu rotace
imAll = create_features(im);

pred = predict(net, imAll);
pred(pred>0)=1;pred(pred<=0)=-1

%% rotace

angle = 360- vec2angle(pred');

imRot = rot90_3D(im, 3, rot(3)/90);
imRot = rot90_3D(imRot, 2, rot(2)/90);
imRot = rot90_3D(imRot, 1, rot(1)/90);


imAll2 = create_features(imRot);


%%
figure
subplot 331
imshow(imOrig(:,:,7),[])
subplot 332
imshow(imOrig(:,:,8),[])
subplot 333
imshow(imOrig(:,:,9),[])

subplot 334
imshow(imAll(:,:,7),[])
subplot 335
imshow(imAll(:,:,8),[])
subplot 336
imshow(imAll(:,:,9),[])

subplot 337
imshow(imAll2(:,:,7),[])
subplot 338
imshow(imAll2(:,:,8),[])
subplot 339
imshow(imAll2(:,:,9),[])
