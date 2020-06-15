clear all
close all
clc

lgraph = create_net;

% name of trained network
name = 'Trained_net_1.mat';


%% datastore
trainData = readtable('C:\Data\Jakubicek\CTDeepRot_data\Datasets\labels.xlsx','ReadVariableNames',false);
ind = [1553:1:size(trainData,1) - (20*24)];
trainData = trainData(ind,1:4);

rng(77)
ind2 = randperm(size(trainData,1));
trainData = trainData(ind2,:);

path = 'C:\Data\Jakubicek\CTDeepRot_data\Datasets\';
imdsTrain = imageDatastore([path 'max_All'],'ReadFcn',@ReaderMultiChannel);
imdsTrain.Files = cellfun(@(x) [path 'max_All\' x '_R4_Ch1.png'], trainData{:,1},'UniformOutput',false);

pathL = 'C:\Data\Jakubicek\CTDeepRot_data\Datasets\labels_mat';
imdsTrainL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid_class,'FileExtensions','.mat');
imdsTrainL.Files = imdsTrainL.Files(ind);
imdsTrainL.Files = imdsTrainL.Files(ind2);

imdsTrainComb = combine(imdsTrain,imdsTrainL);

%%
testData = readtable('C:\Data\Jakubicek\CTDeepRot_data\Datasets\labels.xlsx','ReadVariableNames',false);
ind = [1:1:1552];
testData = testData(ind,1:4);

ind2 = randperm(size(testData,1));
testData = testData(ind2,:);

path = 'C:\Data\Jakubicek\CTDeepRot_data\Datasets\';
imdsTest = imageDatastore([path 'max_All'],'ReadFcn',@ReaderMultiChannel);
imdsTest.Files = cellfun(@(x) [path 'max_All\' x '_R4_Ch1.png'], testData{:,1},'UniformOutput',false);

pathL = 'C:\Data\Jakubicek\CTDeepRot_data\Datasets\labels_mat';
imdsTestL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid_class,'FileExtensions','.mat');
imdsTestL.Files = imdsTestL.Files(ind);
imdsTestL.Files = imdsTestL.Files(ind2);

imdsTestComb = combine(imdsTest,imdsTestL);


%% training
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',1, ...
    'MaxEpochs',4, ...
    'ValidationFrequency',60,...
    'ValidationData', imdsTestComb, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress',...
    'Shuffle','every-epoch',...
    'L2Regularization',0.0002,...
    'InitialLearnRate',0.0001);

%   'ValidationData', imdsTestComb, ...
%%
net = trainNetwork(imdsTrainComb,lgraph,options);

%%
save(['Trained_nets\' name '.mat'],'net','imdsTestComb','imdsTrainComb','options')
 
h = findall(groot, 'Type', 'Figure');
set(h,'units','normalized','outerposition',[0 0 1 1]);
saveas(h,['Trained_nets\' name '.png' ]);

