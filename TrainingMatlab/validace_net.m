%% validace nets

clear all
close all
clc

% load('Trained_nets\Net_1_2.mat')
load('Trained_nets\Net_test.mat')

%% datastore

% trainData = readtable('labels_bin_train.xlsx','ReadVariableNames',false);
% ind = [1:7:3000];
% trainData = trainData(ind,:);
% 
% path = 'C:\Data\Jakubicek\CTDeepRot_data\training\';
% imdsTrain = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
% imdsTrain.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], trainData{:,1},'UniformOutput',false);
% pathL = 'C:\Data\Jakubicek\CTDeepRot_data\training\labels2_mat';
% imdsTrainL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid,'FileExtensions','.mat');
% imdsTrainL.Files = imdsTrainL.Files(ind);
% 
% imdsTrainComb = combine(imdsTrain,imdsTrainL);

% testData = readtable('C:\Data\Jakubicek\CTDeepRot_data\testing\labels.xlsx','ReadVariableNames',false);
% ind = (1:10:5000);
% testData = testData(ind,:);
% 
% path = 'C:\Data\Jakubicek\CTDeepRot_data\testing\';
% imdsTest = imageDatastore([path 'mean_20'],'ReadFcn',@ReaderMultiChannel);
% imdsTest.Files = cellfun(@(x) [path 'mean_20\' x '_R1_Ch1.png'], testData{:,1},'UniformOutput',false);
% pathL = 'C:\Data\Jakubicek\CTDeepRot_data\testing\labels2_mat';
% imdsTestL = imageDatastore([pathL '\'],'ReadFcn',@ReaderValid,'FileExtensions','.mat');
% imdsTestL.Files = imdsTestL.Files(ind);
% imdsTestComb = combine(imdsTest,imdsTestL);

% imdsValid = imdsTestComb.UnderlyingDatastores{1};
% labels = imdsTestComb.UnderlyingDatastores{2}.Files;
% labels = cellfun(@(x) ReaderValid(x), labels,'UniformOutput',false);

imdsValid = imdsTrainComb.UnderlyingDatastores{1};
labels = imdsTrainComb.UnderlyingDatastores{2}.Files;
% labels = cellfun(@(x) ReaderValid(x), labels,'UniformOutput',false);
labels = cellfun(@(x) ReaderValid_class(x), labels,'UniformOutput',false);

%% Prediction

pred = predict(net, imdsValid);

%% statistic

% i=1;ACC=[];PRAH=[];
% for prah = 0:0.01:1
%     ACC(i) = sum( sum( (testData{:,2:7} == (pred<=prah)),2)==6) ./ size(pred,1) *100;
%     PRAH(i) = prah;
%     i=i+1;
% end

GT = cell2mat(labels);
% pred = round(pred);

RMSE = sqrt( sum(((GT - pred).^2),'All')/(numel(GT)) )


% ACC = sum( sum( (GT == (pred)),2)==6) ./ size(pred,1) *100
ACC = sum( sum( (GT == round(pred)),2)==6) ./ size(pred,1) *100


% figure
% plot(PRAH,ACC)


