clc;clear all;close all;


path='Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data';


output_size=128;


file_names=subdir([path filesep '*.mhd']);
file_names={file_names(:).name};


for file_num=1:length(file_names)
    file_num
    
    file_name=file_names{file_num};
    
    file_name_save=file_name;
    file_name_save=replace(file_name_save,'CT_rotation_data',['CT_rotation_data_mat_' num2str(output_size)]);
    file_name_save=replace(file_name_save,'.mhd','.mat');
    [filepath,name,ext] = fileparts(file_name_save);
    mkdir(filepath)
    
    
    [data,info] = load_raw(file_name);
    
    data= imresize3(single(data),[output_size output_size output_size]);
    
    save(file_name_save,'data')
    
    
    
    
    
end









