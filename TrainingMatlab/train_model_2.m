clear all
close all
clc

% lgraph = resnet50;

% load('Nets\Net_1_regr.mat')  % Regresni s dropoutem
% load('Nets\Net_2_regr.mat')  % Regresni bez dropoutu
% load('Nets\Net_2_class.mat')  % Regresni bez dropoutu
% load('Nets\Net_3_regr.mat')  % Nova regresni sit s dropoutem a special regresni vrstvou s ACC
% load('Nets\Net_4_class.mat')  % Nova class sit s dropoutem a special class vrstvou s ACC a one-hot
load('Nets\Net_4_regr.mat')  % Nova regr sit s dropoutem a special regr vrstvou s ACC a one-hot

% reluLayer
% nnet.cnn.layer.TanhLayer

% load('Trained_nets\Net_1_regr_6.mat')
% lgraph = layerGraph(net);

name = 'Net_4_class_4';

InLayer = imageInputLayer([224 224 3],'Name','imageinput','Normalization','zerocenter','DataAugmentation','none');
lgraph = replaceLayer(lgraph,'imageinput',InLayer);

% RegrLayer = customRegresionLayer('regressionoutput');
% lgraph = replaceLayer(lgraph,'classoutput',RegrLayer);
% % % lgraph = connectLayers(lgraph,'fc1000','regressionoutput');
% 
% SML = mySoftMax('softmax');
% lgraph = replaceLayer(lgraph,'softmax',SML);
% 
% ClassLayer = customClassificationLayer('ClassificationOutput');
% lgraph = replaceLayer(lgraph,'regressionoutput',ClassLayer);
% % % lgraph = connectLayers(lgraph,'softmax','ClassificationOutput');

% lgraph = removeLayers(lgraph, {'dropout'});
% lgraph = connectLayers(lgraph,'avg_pool','fc1000');

% lgraph = removeLayers(lgraph, {'fcout','softmax','classoutput'});
% newLayers = [
%     fullyConnectedLayer(2,'Name','fcout','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classoutput')];
% lgraph = addLayers(lgraph,newLayers);
% % lgraph = connectLayers(lgraph,'avg_pool','fcout');
% lgraph = connectLayers(lgraph,'dropout','fcout');


%% datastore

trainData = readtable('C:\Data\Jakubicek\CTDeepRot_data\Datasets\labels.xlsx','ReadVariableNames',false);
% ind = [1:1:128];
% label = cellfun(@(x) codingAngle(x), mat2cell(table2array(trainData(:,2:4)),ones(1,size(trainData,1)),3),'UniformOutput',false);
% label = cell2mat(label')';
% [~,ind]= unique(cat(2,sort(repmat(1:80,1,128))',label), 'rows', 'stable');
% trainData = trainData(ind,1:4);
ind = [1553:1:size(trainData,1) - (20*24)];
trainData = trainData(ind,1:4);

% ind2 = randperm(size(trainData,1));
% trainData = trainData(ind2,:);

path = 'C:\Data\Jakubicek\CTDeepRot_data\Datasets\';
imdsTrain = imageDatastore([path 'max_All'],'ReadFcn',@ReaderMultiChannel);
imdsTrain.Files = cellfun(@(x) [path 'max_All\' x '_R4_Ch1.png'], trainData{:,1},'UniformOutput',false);

pathL = 'C:\Data\Jakubicek\CTDeepRot_data\Datasets\labels_mat';
imdsTrainL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid_class,'FileExtensions','.mat');
imdsTrainL.Files = imdsTrainL.Files(ind);
% imdsTrainL.Files = imdsTrainL.Files(ind2);

imdsTrainComb = combine(imdsTrain,imdsTrainL);

%%
testData = readtable('C:\Data\Jakubicek\CTDeepRot_data\Datasets\labels.xlsx','ReadVariableNames',false);
% label = cellfun(@(x) codingAngle(x), mat2cell(table2array(testData(:,2:4)),ones(1,size(testData,1)),3),'UniformOutput',false);
% label = cell2mat(label')';
% [~,ind]= unique(cat(2,sort(repmat(1:40,1,128))',label), 'rows', 'stable');
ind = [1:1:1552 ];
testData = testData(ind,1:4);

% ind2 = randperm(size(testData,1));
% testData = testData(ind2,:);

path = 'C:\Data\Jakubicek\CTDeepRot_data\Datasets\';
imdsTest = imageDatastore([path 'max_All'],'ReadFcn',@ReaderMultiChannel);
imdsTest.Files = cellfun(@(x) [path 'max_All\' x '_R4_Ch1.png'], testData{:,1},'UniformOutput',false);

pathL = 'C:\Data\Jakubicek\CTDeepRot_data\Datasets\labels_mat';
imdsTestL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid_class,'FileExtensions','.mat');
imdsTestL.Files = imdsTestL.Files(ind);
% imdsTestL.Files = imdsTestL.Files(ind2);

imdsTestComb = combine(imdsTest,imdsTestL);

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
% nnet.cnn.layer.TanhLayer

net = trainNetwork(imdsTrainComb,lgraph,options);

% net = trainNetwork(imdsTrainComb,layerGraph(net),options);


%%
save(['Trained_nets\' name '.mat'],'net','imdsTestComb','imdsTrainComb','options')
 
h = findall(groot, 'Type', 'Figure');
set(h,'units','normalized','outerposition',[0 0 1 1]);
saveas(h,['Trained_nets\' name '.png' ]);

