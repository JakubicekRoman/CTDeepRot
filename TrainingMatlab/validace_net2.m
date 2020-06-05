%% validace nets

clear all
close all
clc

load('Trained_nets\Net_test.mat')

%% datastore

% imdsValid = imdsTestComb.UnderlyingDatastores{1};
% labels = imdsTestComb.UnderlyingDatastores{2}.Files;
% labels = cellfun(@(x) ReaderValid(x), labels,'UniformOutput',false);

imdsValid = imdsTrainComb.UnderlyingDatastores{1};
labels = imdsTrainComb.UnderlyingDatastores{2}.Files;
labels = cellfun(@(x) ReaderValid(x), labels,'UniformOutput',false);

%% Prediction

pred = predict(net, imdsValid);

%% statistic
GT = cell2mat(labels);
% pred = round(pred);
RMSE = sqrt( sum(((GT - pred).^2),'All')/(numel(GT)) )

ACC = sum( sum( (GT == (pred)),2)==6) ./ size(pred,1) *100



