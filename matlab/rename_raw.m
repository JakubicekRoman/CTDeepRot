clc;clear all;close all;
addpath('utils')


file_names=subdir('Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data\*.mhd');

file_names={file_names(:).name};


for file_num=1:length(file_names)
    file_num
    
    filename_in=file_names{file_num};
    
    filename_out=filename_in;
    
    filename_out=replace(filename_out,'Z:\CELL\sdileni_jirina_roman_tom\CT_rotation_data','D:\vicar\tmp_romanovi_rotace\CT_rotation_data_x');

    [Data,Info] = load_raw(filename_in);

    [filepath,name,ext] = fileparts(filename_out);

    Name=['x',name,ext];
    NewPath=filepath;

    mat2raw(Data,NewPath,Name,Info)



end