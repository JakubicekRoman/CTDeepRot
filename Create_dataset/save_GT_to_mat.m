clear all
close all
clc

% path_data = ['\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\training'];
path_data = ['\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\testing'];

% path_save = '\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\training\labels_mat';
path_save = '\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\testing\labels_mat';
mkdir(path_save)

lbl = readcell([path_data '\labels_bin.xlsx']);

for i = 1:size(lbl,1)
    GT = [lbl{i,2:8}];
    save([path_save '\' lbl{i,1} '.mat'],'GT');
%     writetable(T, 'MyFile.txt')
%     fid=fopen([path_save '\' lbl{i,1} '.txt'],'w');
%    fprintf(fid, '%f \n', [GT]');
%     fclose(fid);true
end