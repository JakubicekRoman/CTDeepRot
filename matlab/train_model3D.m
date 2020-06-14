clear all;close all force;clc;
addpath('utils')

data_path='../../CT_rotation_data_mat_128';

% names=subdir([data_path '/*.mat']);
% names={names(:).name};
% for name =names
%     delete(name{1})
% end


rot_table=readtable('utils/rot_dict_unique.csv');
unique_rots=rot_table{:,1:3};

data_table=readtable([data_path filesep 'ListOfData.xlsx']);
files={};
for k=1:size(data_table,1)
    tmp=data_table{k,1};
    tmp=split(tmp,'\');
    tmp2=data_table{k,2};
    tmp2=replace(tmp2{1},'.mhd','');
    files=[files, [data_path filesep tmp{end} filesep tmp2]];
end



names={};
for file_num=1:length(files)
    for flip = [0,1]
       for rot_num=1:size(unique_rots,1)
           
           tmp=[files{file_num} '_' num2str(unique_rots(rot_num,1),'%03.f') '_' num2str(unique_rots(rot_num,2),'%03.f') '_' num2str(unique_rots(rot_num,3),'%03.f') '_' num2str(flip,'%03.f') '.mat'];
           names=[names tmp];
           aa=1;
           if ~isfile(tmp)
                save(tmp,'aa')
           end
       end
    end
end




names_train=names(1:round(0.8*length(names)));
names_test=names(round(0.8*length(names))+1:end-20);


imdsTrain = imageDatastore(names_train,'FileExtensions','.mat','ReadFcn',@ReadData3D);
imdsTrainL = imageDatastore(names_train,'FileExtensions','.mat','ReadFcn',@ReadData3D_lbl);
imdsTrainComb = combine(imdsTrain,imdsTrainL);


imdsTest = imageDatastore(names_test,'FileExtensions','.mat','ReadFcn',@ReadData3D);
imdsTestL = imageDatastore(names_test,'FileExtensions','.mat','ReadFcn',@ReadData3D_lbl);
imdsTestComb = combine(imdsTrain,imdsTestL);



lgraph = simple_net();



bs=8;
vf=length(imdsTrain.Files)/bs/2;
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',30, ...
    'MaxEpochs',70, ...
    'ValidationFrequency',vf,...
    'ValidationData', imdsTestComb, ...
    'MiniBatchSize',bs, ...
    'Plots','training-progress',...
    'Shuffle','every-epoch',...
    'L2Regularization',1e-6,...
    'InitialLearnRate',0.001);


net = trainNetwork(imdsTrainComb,lgraph,options);



%%
mkdir('../../Trained_nets')
save(['../../Trained_nets\' name '.mat'],'net','imdsTestComb','imdsTrainComb','options')

h = findall(groot, 'Type', 'Figure');
set(h,'units','normalized','outerposition',[0 0 1 1]);
saveas(h,['../../Trained_nets\' name '.png' ]);