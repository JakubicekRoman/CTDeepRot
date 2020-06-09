clear all;close all force;clc



train_data_path='D:\jakubicek\Rot_detection\data_3d_128_mat\Data_raw_train';
test_data_path='D:\jakubicek\Rot_detection\data_3d_128_mat\Data_raw_test';




%% datastore



imdsTrain = imageDatastore(train_data_path,'FileExtensions','.mat','ReadFcn',@ReadData3D);
files=imdsTrain.Files;
names={};
for file_num=1:length(files)
    for flip = [0,1]
        unique_rots={};
        for a=[0,90,180,270]
            for b=[0,90,180,270]
                for c=[0,90,180,270]
                    
                    rot_vec = codingAngle([a,b,c]);
                    
                    new=sum(cellfun(@(x) all(x==rot_vec),unique_rots))==0;
                    
                    if new
                        tmp=[files{file_num} '_' num2str(a,'%03.f') '_' num2str(b,'%03.f') '_' num2str(c,'%03.f') '_' num2str(flip,'%03.f') '.tmp'];
                        names=[names tmp];
                        aa=1;
                        save(tmp,'aa')
                        unique_rots=[unique_rots,rot_vec];
                    end
                end
            end
        end
    end
end
imdsTrain.Files=names;


imdsTrainL = imageDatastore(train_data_path,'FileExtensions','.mat','ReadFcn',@ReadData3D_lbl);
imdsTrainL.Files = names;
for name =names
    delete(name{1})
end

imdsTrainComb = combine(imdsTrain,imdsTrainL);




%%
imdsTest = imageDatastore(test_data_path,'FileExtensions','.mat','ReadFcn',@ReadData3D);
files=imdsTest.Files;
names={};
for file_num=1:length(files)
    for flip = [0,1]
        unique_rots={};
        for a=[0,90,180,270]
            for b=[0,90,180,270]
                for c=[0,90,180,270]
                    
                    rot_vec = codingAngle([a,b,c]);
                    
                    new=sum(cellfun(@(x) all(x==rot_vec),unique_rots))==0;
                    
                    if new
                        tmp=[files{file_num} '_' num2str(a,'%03.f') '_' num2str(b,'%03.f') '_' num2str(c,'%03.f') '_' num2str(flip,'%03.f') '.tmp'];
                        names=[names tmp];
                        aa=1;
                        save(tmp,'aa')
                        unique_rots=[unique_rots,rot_vec];
                    end
                end
            end
        end
    end
end
imdsTest.Files=names;


imdsTestL = imageDatastore(test_data_path,'FileExtensions','.mat','ReadFcn',@ReadData3D_lbl);
imdsTestL.Files = names;
for name =names
    delete(name{1})
end

imdsTestComb = combine(imdsTest,imdsTestL);


lgraph = simple_net();



%% training
% options = trainingOptions('adam', ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.5, ...
%     'LearnRateDropPeriod',2, ...
%     'MaxEpochs',25, ...
%     'ValidationFrequency',15,...
%     'ValidationData', imdsTestComb, ...
%     'MiniBatchSize',64, ...
%     'Plots','training-progress',...
%     'Shuffle','every-epoch',...
%     'L2Regularization',0.0002,...
%     'InitialLearnRate',0.000008);

%% training

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

%   'ValidationData', imdsTestComb, ...

net = trainNetwork(imdsTrainComb,lgraph,options);
% % net = trainNetwork(imdsTrain,lgraph, options);
% net = trainNetwork(imdsTrainComb,layerGraph(net),options);


%%
mkdir('../../Trained_nets')
save(['../../Trained_nets\' name '.mat'],'net','imdsTestComb','imdsTrainComb','options')

h = findall(groot, 'Type', 'Figure');
set(h,'units','normalized','outerposition',[0 0 1 1]);
saveas(h,['Trained_nets\' name '.png' ]);

