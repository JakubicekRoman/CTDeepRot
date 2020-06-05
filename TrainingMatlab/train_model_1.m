clear all
close all
clc

% lgraph = resnet50;

% load('Nets\Net_1.mat')  % Regresni s dropoutem
load('Nets\Net_2.mat')  % Regresni bez dropoutu

name = 'Net_test';

%% datastore

trainData = readtable('labels_bin_train.xlsx','ReadVariableNames',false);
ind = [1:30:2000];
trainData = trainData(ind,:);

path = 'C:\Data\Jakubicek\CTDeepRot_data\training\';
imdsTrain = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
imdsTrain.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], trainData{:,1},'UniformOutput',false);
pathL = 'C:\Data\Jakubicek\CTDeepRot_data\training\labels2_mat';
imdsTrainL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid,'FileExtensions','.mat');
imdsTrainL.Files = imdsTrainL.Files(ind);

imdsTrainComb = combine(imdsTrain,imdsTrainL);

testData = readtable('C:\Data\Jakubicek\CTDeepRot_data\testing\labels.xlsx','ReadVariableNames',false);
ind = [1:50:1000];
testData = testData(ind,:);

path = 'C:\Data\Jakubicek\CTDeepRot_data\testing\';
imdsTest = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
imdsTest.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], testData{:,1},'UniformOutput',false);
pathL = 'C:\Data\Jakubicek\CTDeepRot_data\testing\labels2_mat';
imdsTestL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid,'FileExtensions','.mat');
imdsTestL.Files = imdsTestL.Files(ind);

imdsTestComb = combine(imdsTest,imdsTestL);

%% training
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',15, ...
    'MaxEpochs',30, ...
    'ValidationData', imdsTestComb, ...
    'ValidationFrequency',10,...
    'MiniBatchSize',8, ...
    'Plots','training-progress',...
    'Shuffle','every-epoch',...
    'L2Regularization',0.00001,...
    'InitialLearnRate',0.0001);


net = trainNetwork(imdsTrainComb,lgraph,options);

save(['Trained_nets\' name '.mat'],'net','imdsTestComb','imdsTrainComb')
 
h = findall(groot, 'Type', 'Figure');
set(h,'units','normalized','outerposition',[0 0 1 1]);
saveas(h,['Trained_nets\' name '.png' ]);
