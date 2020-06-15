%% finalni program pro rotaci

clear all
close all
clc


load('Trained_nets\Net_4_class_3.mat','net')

% 
% path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\Korez\' 'case26_original.mhd' ];
path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\DIR\raw\' 'DIR_Data_3.mhd' ];
% path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\Dataset_3_CT_Glocker_242_pat_vyber\Data_raw\' '4557469.mhd' ];
% path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\Dataset_3_CT_Glocker_242_pat_vyber\Data_raw\' '4574668.mhd' ];
% path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\vc prone and supine 3 cases\raw\' 'colon_sups_3.mhd' ];

imO = load_raw(path_raw);

% imOrig = create_features(imO);

ROT = [0,0,0 1;0,0,90 2;0,0,180 3;0,0,270 4;0,90,0 5;0,90,90 6;0,90,180 7;0,90,270 8;0,180,0 9;0,180,90 10;0,180,180 11;0,180,270 12;0,270,0 13;0,270,90 14;0,270,180 15;0,270,270 16;90,0,0 17;90,0,90 18;90,0,180 19;90,0,270 20;90,180,0 21;90,180,90 22;90,180,180 23;90,180,270 24];


for r = 1:24
    rot = ROT(r,1:3);
% im = imrotate3(im,rot(1),[1,0,0],'nearest');
im = rot90_3D(imO, 1, rot(1)/90);
% im = imrotate3(im,rot(2),[0,1,0],'nearest'); % rotY - rotace podle X osy
im = rot90_3D(im, 2, rot(2)/90);
% im = imrotate3(im,rot(3),[0,0,1],'nearest'); % rotZ - rotace AXIAL    
im = rot90_3D(im, 3, rot(3)/90);

% codingAngle(rot)


%% nalezeni uhlu rotace
imAll = create_features(im);
% imAll(:,:,1:3) = imAll(:,:,7:9);
% imAll(:,:,4:6) = imAll(:,:,7:9);

pred = predict(net, imAll);
max(pred)
pred(pred==max(pred))=1;
pred(pred~=1)=0;


%% rotace

angle = 360 - decodingAngle(pred);
angle(angle==360)=0;

imRot = rot90_3D(im, 3, angle(3)/90);
imRot = rot90_3D(imRot, 2, angle(2)/90);
imRot = rot90_3D(imRot, 1, angle(1)/90);

imAll2 = create_features(imRot);


%%
figure(1)
% subplot 331
% imshow(imOrig(:,:,1),[])
% subplot 332
% imshow(imOrig(:,:,2),[])
% subplot 333
% imshow(imOrig(:,:,3),[])

subplot 334
imshow(imAll(:,:,1),[])
subplot 335
imshow(imAll(:,:,2),[])
subplot 336
imshow(imAll(:,:,3),[])

subplot 337
imshow(imAll2(:,:,1),[])
subplot 338
imshow(imAll2(:,:,2),[])
subplot 339
imshow(imAll2(:,:,3),[])

pause(1)
end
