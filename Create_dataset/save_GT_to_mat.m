clear all
close all
clc

path_data = ['C:\Data\Jakubicek\CTDeepRot_data\Datasets'];
% path_data = ['C:\Data\Jakubicek\CTDeepRot_data\testing'];

path_save = 'C:\Data\Jakubicek\CTDeepRot_data\Datasets\labels_mat';
% path_save = 'C:\Data\Jakubicek\CTDeepRot_data\testing\labels_mat';
mkdir(path_save)

lbl = readcell([path_data '\labels.xlsx']);

rot = [0,0,0 1;0,0,90 2;0,0,180 3;0,0,270 4;0,90,0 5;0,90,90 6;0,90,180 7;0,90,270 8;0,180,0 9;0,180,90 10;0,180,180 11;0,180,270 12;0,270,0 13;0,270,90 14;0,270,180 15;0,270,270 16;90,0,0 17;90,0,90 18;90,0,180 19;90,0,270 20;90,180,0 21;90,180,90 22;90,180,180 23;90,180,270 24];

% classificationLayer
% crossentropyex

for i = 1:size(lbl,1)
   
    GT = [lbl{i,2:4}];   
    [~,p] = min(sum(abs((rot(:,1:3)) - (GT)),2));
    
%     % one-hot
%     GT = zeros(24,1);
%     GT(p) = 1;
    
    % class
    GT = p;
    
%     GTb([1,3,5])=sind(GT);
%     GTb([2,4,6])=cosd(GT);
%     GT=GTb;
    save([path_save '\' lbl{i,1} '.mat'],'GT');
%     fid=fopen([path_save '\' lbl{i,1} '.txt'],'w');
%    fprintf(fid, '%f \n', [GTb]');
%     fclose(fid);true
end