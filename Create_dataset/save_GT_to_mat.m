clear all
close all
clc

% path_data = ['\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\training'];
path_data = ['\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\testing'];

% path_save = '\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\training\labels2_mat';
path_save = '\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\testing\labels2_mat';
mkdir(path_save)

lbl = readcell([path_data '\labels.xlsx']);

for i = 1:size(lbl,1)
    GT = [lbl{i,2:4}];
    GTb([1,3,5])=sind(GT);
    GTb([2,4,6])=cosd(GT);
    GT=GTb;
    save([path_save '\' lbl{i,1} '.mat'],'GT');
%     fid=fopen([path_save '\' lbl{i,1} '.txt'],'w');
%    fprintf(fid, '%f \n', [GTb]');
%     fclose(fid);true
end