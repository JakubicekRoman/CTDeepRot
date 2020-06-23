clear all;close all force;clc;
addpath('utils')

%dbstop if error
%dbclear if error


load('../../Trained_nets/net2d_9775.mat')


data_path='Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data';





rot_table=readtable('utils/rot_dict_unique.csv');
unique_rots=rot_table{:,1:3};

data_table=readtable([data_path filesep 'ListOfData.xlsx']);
files={};
for k=1:size(data_table,1)
    tmp=data_table{k,1};
    tmp=split(tmp,'\');
    tmp2=data_table{k,2};
    tmp2=replace(tmp2{1},'.mhd','');
    files=[files, [data_path filesep tmp{end} filesep tmp2]];
end



files=files(round(0.8*length(files))+1:end);


difs=[];
psts=[];
names={};
rotations_gt={};
rotations_predicted={};

for num_file=1:length(files)
    num_file
    
    
    
    file=files{num_file};
    
    data=single(load_raw([file '.mhd']));
    
    data=imresize3(data,[224,224,224]);
    
    for a=[0,90,180,270]
        for b=[0,90,180,270]
            for c=[0,90,180,270]
                
                rot=[a,b,c];
                data_rot=rotate_3d(data,rot);
                
                [imMean,imMax,imStd]=get_2d_feature_imgs(data_rot);
                
                data_2d=cat(3,imMean,imMax,imStd);
                
                data_2d=single(data_2d)/255-0.5;
                
                res=predict(net,data_2d);
                
                
                [mres,ind]=max(res);
                
                pred_rot=unique_rots(ind,:);
                
                
                
                data_fix=rotate_3d_inverse(data_rot,pred_rot);
                
                
                
                tmp=abs(data-data_fix);
                
                dif=sum(tmp(:))
                psts=[psts mres];
                difs=[difs,dif];
                names=[names,file];
                rotations_gt=[rotations_gt,rot];
                rotations_predicted=[rotations_predicted,pred_rot];
                
                
%                 figure(1)
%     
%                 subplot(3,3,1)
%                 imshow(squeeze(mean(data,1)),[])
%                 subplot(3,3,2)
%                 imshow(squeeze(mean(data,2)),[])
%                 subplot(3,3,3)
%                 imshow(squeeze(mean(data,3)),[])
% 
%                 subplot(3,3,4)
%                 imshow(squeeze(mean(data_rot,1)),[])
%                 subplot(3,3,5)
%                 imshow(squeeze(mean(data_rot,2)),[])
%                 subplot(3,3,6)
%                 imshow(squeeze(mean(data_rot,3)),[])
% 
%                 subplot(3,3,7)
%                 imshow(squeeze(mean(data_fix,1)),[])
%                 subplot(3,3,8)
%                 imshow(squeeze(mean(data_fix,2)),[])
%                 subplot(3,3,9)
%                 imshow(squeeze(mean(data_fix,3)),[])
                
                
                
                
            end
        end
    end
    
end


mean(difs==0)


save('results_2d.mat','difs','names','rotations_gt','rotations_predicted','psts')



