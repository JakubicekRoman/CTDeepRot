clc;clear all;close all;
addpath('utils')

path='Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data';


file_names = subdir([path '/*.mhd']);
file_names={file_names(:).name};


for file_num = 1:length(file_names)
    file_num
    
    file_name=file_names{file_num};
    
    file_names_save=file_name;
    file_names_save=replace(file_names_save,'CT_rotation_data','CT_rotation_data_2D');
    file_names_save=replace(file_names_save,'.mhd','');
    
    
    [filepath,name,ext] = fileparts(file_names_save);
    
    mkdir(filepath)
    
    data = load_raw(file_name);

    [imMean,imMax,imStd]=get_2d_feature_imgs(data);
    
    for dim=1:3
        imwrite(imMean(:,:,dim),[file_names_save '_mean_' num2str(dim) '.png'])
    end
    for dim=1:3
        imwrite(imMax(:,:,dim),[file_names_save '_max_' num2str(dim) '.png'])
    end
    for dim=1:3
        imwrite(imStd(:,:,dim),[file_names_save '_std_' num2str(dim) '.png'])
    end
    
end
