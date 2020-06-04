clear all
close all
clc

% path_data = ['\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\training'];
path_data = ['\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\testing'];

lbl = readcell([path_data '\labels_bin.xlsx']);

D = dir([path_data '\**\*.png']);

name = {[D(1).folder '\' D(1).name]};
anot = table(name);
for k = 1:7
    anot = [anot table(true, 'VariableNames', {['Var' num2str(k)]})];
end

for i = 1:size(D,1)
    anot{i,1} = {[D(i).folder '\' D(i).name]};
    ind = ceil(strfind([lbl{:}],D(i).name(1:13))/13);
    anot{i,2:8} = lbl{ind,2:8};
end

% writetable(anot,'train_dataset_001.xlsx')
writetable(anot,'test_dataset_001.xlsx')
