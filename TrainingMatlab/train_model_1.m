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

load('Net_1.mat')

%% datastore
pathL = '\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\training\labels_mat';

trainData = readtable('labels_bin_train.xlsx','ReadVariableNames',false);
trainData = trainData(1:3,:);

path = '\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\training\';
imdsTrain = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
imdsTrain.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], trainData{:,1},'UniformOutput',false);

imdsTrainL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid,'FileExtensions','.mat');
imdsTrainL.Files = imdsTrainL.Files(1:3);

imdsTrainComb = combine(imdsTrain,imdsTrainL);

testData = readtable('labels_bin_test.xlsx','ReadVariableNames',false);
testData = testData(1:3,:);

path = '\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\testing\';
imdsTest = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
imdsTest.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], testData{:,1},'UniformOutput',false);

imdsTestL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid,'FileExtensions','.mat');
imdsTestL.Files = imdsTestL.Files(1:3);

imdsTrainComb = combine(imdsTest,imdsTestL);

%% training
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',20, ...
    'MaxEpochs',50, ...
    'ValidationData', imdsTrainComb, ...
    'MiniBatchSize',8, ...
    'Plots','training-progress',...
    'Shuffle','never',...
    'InitialLearnRate',0.0001);


net = trainNetwork(imdsTrainComb,lgraph,options);

% %     'ValidationData', testData, ...
