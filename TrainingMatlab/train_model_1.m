clear all
close all
clc

% lgraph = resnet50;
% 
% lgraph = layerGraph(lgraph.Layers);
% lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});
% 
% newLayers = [
%     fullyConnectedLayer(7,'Name','fc1000','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','classoutput')];
% 
% lgraph = addLayers(lgraph,newLayers);
% lgraph = connectLayers(lgraph,'avg_pool','fc1000');
% 
% newLayers = imageInputLayer([224,224,18],'Name','input');
% lgraph = replaceLayer(lgraph,'input_1',newLayers);
% 
% newLayer = convolution2dLayer([7,7],64,'Stride',2,'Padding',[3,3],'Name','conv1' );
% lgraph = replaceLayer(lgraph,'conv1',newLayer);

load('Net_3.mat')

%% datastore

trainData = readtable('labels_bin_train.xlsx','ReadVariableNames',false);
trainData = trainData(1:10:10000,:);

path = 'D:\jakubicek\Rot_detection\data\training\';
imdsTrain = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
imdsTrain.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], trainData{:,1},'UniformOutput',false);
pathL = 'D:\jakubicek\Rot_detection\data\training\labels_mat';
imdsTrainL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid,'FileExtensions','.mat');
imdsTrainL.Files = imdsTrainL.Files(1:10:10000);

imdsTrainComb = combine(imdsTrain,imdsTrainL);

testData = readtable('labels_bin_test.xlsx','ReadVariableNames',false);
testData = testData(1:40:3000,:);

path = 'D:\jakubicek\Rot_detection\data\testing\';
imdsTest = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
imdsTest.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], testData{:,1},'UniformOutput',false);
pathL = 'D:\jakubicek\Rot_detection\data\testing\labels_mat';
imdsTestL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid,'FileExtensions','.mat');
imdsTestL.Files = imdsTestL.Files(1:40:3000);

imdsTestComb = combine(imdsTest,imdsTestL);

%% training
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'ValidationFrequency',50, ...
    'MaxEpochs',40, ...
    'L2Regularization', 0.01, ...
    'ValidationData', imdsTestComb, ...
    'MiniBatchSize',16, ...
    'Plots','training-progress',...
    'Shuffle','every-epoch',...
    'InitialLearnRate',0.01);


net = trainNetwork(imdsTrainComb,lgraph,options);

