clear all
close all
clc

% path_data = ['C:\Data\Jakubicek\CTDeepRot_data\training'];
path_data = ['C:\Data\Jakubicek\CTDeepRot_data\testing'];

% path_save = 'C:\Data\Jakubicek\CTDeepRot_data\training\labels2_mat';
path_save = 'C:\Data\Jakubicek\CTDeepRot_data\testing\labels2_mat';
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