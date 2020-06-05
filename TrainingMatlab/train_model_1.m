clear all
close all
clc

% lgraph = resnet50;

<<<<<<< HEAD
% load('Nets\Net_1.mat')  % Regresni s dropoutem
load('Nets\Net_2.mat')  % Regresni bez dropoutu

name = 'Net_test';
=======
load('Net_3.mat')
>>>>>>> 99be23a5d877ff86e57bde57463ac74726901c7e

%% datastore

trainData = readtable('labels_bin_train.xlsx','ReadVariableNames',false);
<<<<<<< HEAD
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
=======
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
>>>>>>> 99be23a5d877ff86e57bde57463ac74726901c7e

imdsTestComb = combine(imdsTest,imdsTestL);

%% training
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
<<<<<<< HEAD
    'LearnRateDropPeriod',15, ...
    'MaxEpochs',30, ...
    'ValidationData', imdsTestComb, ...
    'ValidationFrequency',10,...
    'MiniBatchSize',8, ...
    'Plots','training-progress',...
    'Shuffle','every-epoch',...
    'L2Regularization',0.00001,...
    'InitialLearnRate',0.0001);
=======
    'LearnRateDropPeriod',5, ...
    'ValidationFrequency',50, ...
    'MaxEpochs',40, ...
    'L2Regularization', 0.01, ...
    'ValidationData', imdsTestComb, ...
    'MiniBatchSize',16, ...
    'Plots','training-progress',...
    'Shuffle','every-epoch',...
    'InitialLearnRate',0.01);
>>>>>>> 99be23a5d877ff86e57bde57463ac74726901c7e


net = trainNetwork(imdsTrainComb,lgraph,options);

<<<<<<< HEAD
save(['Trained_nets\' name '.mat'],'net','imdsTestComb','imdsTrainComb')
 
h = findall(groot, 'Type', 'Figure');
set(h,'units','normalized','outerposition',[0 0 1 1]);
saveas(h,['Trained_nets\' name '.png' ]);
=======
>>>>>>> 99be23a5d877ff86e57bde57463ac74726901c7e
