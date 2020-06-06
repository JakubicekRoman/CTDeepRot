clear all
close all
clc

% lgraph = resnet50;

% load('Nets\Net_1.mat')  % Regresni s dropoutem
% load('Nets\Net_2_regr.mat')  % Regresni bez dropoutu
load('Nets\Net_2_class.mat')  % Regresni bez dropoutu

name = 'Net_test_Class';

lgraph = removeLayers(lgraph, {'fcout','softmax','classoutput'});
newLayers = [
    fullyConnectedLayer(4,'Name','fcout','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
% lgraph = connectLayers(lgraph,'avg_pool','fcout');
lgraph = connectLayers(lgraph,'dropout','fcout');


%% datastore

trainData = readtable('C:\Data\Jakubicek\CTDeepRot_data\training\labels.xlsx','ReadVariableNames',false);
ind = [1:5:10000];
trainData = trainData(ind,1:2);
label = categorical(trainData{:,2});

path = 'C:\Data\Jakubicek\CTDeepRot_data\training\';
imdsTrain = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
imdsTrain.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], trainData{:,1},'UniformOutput',false);
% pathL = 'C:\Data\Jakubicek\CTDeepRot_data\training\labels2_mat';
% imdsTrainL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid_class,'FileExtensions','.mat');
% imdsTrainL.Files = imdsTrainL.Files(ind);
imdsTrain.Labels = label;
% 
% imdsTrainComb = combine(imdsTrain,imdsTrainL);
% 
testData = readtable('C:\Data\Jakubicek\CTDeepRot_data\testing\labels.xlsx','ReadVariableNames',false);
ind = [1:20:5000];
testData = testData(ind,1:2);
label = categorical(testData{:,2});
% 
path = 'C:\Data\Jakubicek\CTDeepRot_data\testing\';
imdsTest = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
imdsTest.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], testData{:,1},'UniformOutput',false);
% pathL = 'C:\Data\Jakubicek\CTDeepRot_data\testing\labels2_mat';
% imdsTestL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid_class,'FileExtensions','.mat');
% imdsTestL.Files = imdsTestL.Files(ind);
imdsTest.Labels = label;
% 
% imdsTestComb = combine(imdsTest,imdsTestL);

%% training
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',20, ...
    'ValidationFrequency',40,...
    'ValidationData', imdsTest, ...
    'MiniBatchSize',32, ...
    'Plots','training-progress',...
    'Shuffle','every-epoch',...
    'L2Regularization',0.01,...
    'InitialLearnRate',0.00001);

%   'ValidationData', imdsTestComb, ...

% net = trainNetwork(imdsTrainComb,lgraph,options);
net = trainNetwork(imdsTrain,lgraph, options);


%%
save(['Trained_nets\' name '.mat'],'net','imdsTestComb','imdsTrainComb')
 
h = findall(groot, 'Type', 'Figure');
set(h,'units','normalized','outerposition',[0 0 1 1]);
saveas(h,['Trained_nets\' name '.png' ]);

