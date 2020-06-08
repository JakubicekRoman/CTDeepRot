%% rename
clear all

% D = dir('C:\Data\Jakubicek\CTDeepRot_data\training\mean_20\D02*.png');
% D = dir('C:\Data\Jakubicek\CTDeepRot_data\training\mean_All\D02*.png');
% D = dir('C:\Data\Jakubicek\CTDeepRot_data\training\max_40\D02*.png');
% D = dir('C:\Data\Jakubicek\CTDeepRot_data\training\max_All\D02*.png');
% D = dir('C:\Data\Jakubicek\CTDeepRot_data\training\std_40\D02*.png');
D = dir('C:\Data\Jakubicek\CTDeepRot_data\training\std_All\D02*.png');

for i=1:length(D)
    nameO = D(i).name;
    nameN =  strrep(D(i).name,'D02','D01');
    movefile([D(i).folder '\' nameO],[D(i).folder '\' nameN])
%     delete([D(i).folder '\' nameO])
end