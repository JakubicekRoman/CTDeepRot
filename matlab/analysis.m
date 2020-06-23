clc;clear all;close all;
addpath('utils')

% load('results_2d.mat')
load('results_3d.mat')

mean(difs==0)


% rot_table=readtable('utils/rot_dict_unique.csv');

remove=strcmp(,'Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data\VerSe2020_train\VerSe20_0112';




spatne=difs>0;


spatne_names=names(spatne);

spatne_names_u=unique(spatne_names);
kolik_spatne=zeros(1,length(spatne_names_u));
for k = 1:length(spatne_names_u)
    
    
    
    
    kolik_spatne(k)=sum(strcmp(spatne_names,spatne_names_u{k}));
    
end

kolik_spatne;
spatne_names_u';


for k=length(spatne_names_u)
    
    file=spatne_names_u{k};
    data=single(load_raw([file '.mhd']));
    
    
    
    data=imresize3(data,[224,224,224]);
    
    [imMean,imMax,imStd]=get_2d_feature_imgs(data);
    
    tmp=imStd;
        figure()
    subplot(3,3,1)
    imshow(tmp(:,:,1),[])
    title(kolik_spatne(k))
    subplot(3,3,2)
    imshow(tmp(:,:,2),[])
    subplot(3,3,3)
    imshow(tmp(:,:,3),[])

    
%     figure()
%     subplot(3,3,1)
%     imshow(squeeze(mean(data,1)),[])
%     title(kolik_spatne(k))
%     subplot(3,3,2)
%     imshow(squeeze(mean(data,2)),[])
%     subplot(3,3,3)
%     imshow(squeeze(mean(data,3)),[])

    
end










