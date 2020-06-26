clc;clear all;close all;
addpath('utils')

% load('results_2d.mat')
% load('results_3d.mat')

save_folder_name='../../d2';
mkdir(save_folder_name)




load('D:\vicar\tmp_romanovi_rotace\CTDeepRot\python/results_2d.mat')
% load('D:\vicar\tmp_romanovi_rotace\CTDeepRot\python/results_3d.mat')
names={};
for k=1:size(file_names_all,1)
    names=[names,replace(file_names_all(k,:),' ','')];
end
rotations_gt=rots_gt;
rotations_predicted=squeeze(rots_res);




remove=~strcmp(names,'D:\vicar\tmp_romanovi_rotace\CT_rotation_data_x\VerSe2020_train\xVerSe20_0112.mhd.mhd');
difs=difs(remove);
psts=psts(remove);
names=names(remove);
rotations_gt=rotations_gt(remove,:);
rotations_predicted=rotations_predicted(remove,:);


remove=~strcmp(names,'Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data\VerSe2020_train\VerSe20_0112');
difs=difs(remove);
psts=psts(remove);
names=names(remove);
rotations_gt=rotations_gt(remove,:);
rotations_predicted=rotations_predicted(remove,:);




rot_table=readtable('utils/rot_dict_unique.csv');
rot_table=rot_table{:,1:3};
rot_table = num2cell(rot_table,2);


remove=zeros(1,length(names));
for t=1:length(rot_table)
    for n=1:length(names)
        if all(rot_table{t}==rotations_gt(n,:))
            remove(n)=1;
        end
    end
end

remove=remove>0;
difs=difs(remove);
psts=psts(remove);
names=names(remove);
rotations_gt=rotations_gt(remove,:);
rotations_predicted=rotations_predicted(remove,:);






mean(difs==0)



spatne=difs>0;







uhly_spatne=rotations_gt(spatne,:);




spatne_names=names(spatne);

uhly_spatne
spatne_names'


[A,ia,ic] = unique(uhly_spatne,'rows');



spatne_names_u=unique(spatne_names);
kolik_spatne=zeros(1,length(spatne_names_u));
for k = 1:length(spatne_names_u)
    
    
    
    
    kolik_spatne(k)=sum(strcmp(spatne_names,spatne_names_u{k}));
    
end

kolik_spatne
spatne_names_u'




for k=1:length(spatne_names_u)
    
    file=spatne_names_u{k};

%     data=single(load_raw([file '.mhd']));
    data=single(load_raw([file]));
    
    [filepath,file,ext] = fileparts(file);
    
    imwrite(uint8(255*mat2gray(squeeze(mean(data,1)))),[save_folder_name '/' file 'mean1.png'] )
    imwrite(uint8(255*mat2gray(squeeze(mean(data,2)))),[save_folder_name '/' file 'mean2.png'] )
    imwrite(uint8(255*mat2gray(squeeze(mean(data,3)))),[save_folder_name '/' file 'mean3.png'] )
    
    imwrite(uint8(255*mat2gray(squeeze(max(data,[],1)))),[save_folder_name '/' file 'max1.png'] )
    imwrite(uint8(255*mat2gray(squeeze(max(data,[],2)))),[save_folder_name '/' file 'max2.png'] )
    imwrite(uint8(255*mat2gray(squeeze(max(data,[],3)))),[save_folder_name '/' file 'max3.png'] )
    
    imwrite(uint8(255*mat2gray(squeeze(std(data,[],1)))),[save_folder_name '/' file 'std1.png'] )
    imwrite(uint8(255*mat2gray(squeeze(std(data,[],2)))),[save_folder_name '/' file 'std2.png'] )
    imwrite(uint8(255*mat2gray(squeeze(std(data,[],3)))),[save_folder_name '/' file 'std3.png'] )
    
    
    data=imresize3(data,[224,224,224]);
    
    [imMean,imMax,imStd]=get_2d_feature_imgs(data);
    
    
    
    
    
%     tmp=imStd;
%        figure()
%     subplot(3,3,1)
%     imshow(tmp(:,:,1),[])
%     title(kolik_spatne(k))
%     subplot(3,3,2)
%     imshow(tmp(:,:,2),[])
%     subplot(3,3,3)
%     imshow(tmp(:,:,3),[])

    
%     figure()
%     subplot(3,3,1)
%     imshow(squeeze(mean(data,1)),[])
%     title(kolik_spatne(k))
%     subplot(3,3,2)
%     imshow(squeeze(mean(data,2)),[])
%     subplot(3,3,3)
%     imshow(squeeze(mean(data,3)),[])

    
end










