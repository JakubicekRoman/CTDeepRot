clc;clear all;close all;


path='D:\jakubicek\Rot_detection\data_3d';


output_size=128;


file_names=subdir([path filesep '*.mhd']);
file_names={file_names(:).name};


for file_num=1:length(file_names)
    file_num
    
    file_name=file_names{file_num};
    
    file_name_save=file_name;
    file_name_save=replace(file_name_save,'data_3d',['data_3d_' num2str(output_size) '_mat']);
    file_name_save=replace(file_name_save,'.mhd','.mat');
    [filepath,name,ext] = fileparts(file_name_save);
    mkdir(filepath)
    
    
    [data,info] = load_raw(file_name);
    
    data= imresize3(single(data),[output_size output_size output_size]);
    
    save(file_name_save,'data')
    
    
    
    
    
end









