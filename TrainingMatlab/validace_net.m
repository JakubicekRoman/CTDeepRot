%% validace nets

clear all
% close all
clc

% load('Trained_nets\Net_1_2.mat')
load('Trained_nets\Net_1_regr_8.mat')

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

imdsValid1 = imdsTrainComb.UnderlyingDatastores{1};
labels1 = imdsTrainComb.UnderlyingDatastores{2}.Files;
% labels = cellfun(@(x) ReaderValid(x), labels,'UniformOutput',false);
labels1 = cellfun(@(x) ReaderValid(x), labels1,'UniformOutput',false);

imdsValid2 = imdsTestComb.UnderlyingDatastores{1};
labels2 = imdsTestComb.UnderlyingDatastores{2}.Files;
% labels = cellfun(@(x) ReaderValid(x), labels,'UniformOutput',false);
labels2 = cellfun(@(x) ReaderValid(x), labels2,'UniformOutput',false);

%% Prediction

pred1 = predict(net, imdsValid1);
pred2 = predict(net, imdsValid2);

%% statistic

GT = cell2mat(labels1);
i=1;ACC1=[];PRAH1=[];
for prah = -1:0.01:1
    Pred = pred1;Pred(pred1<prah)=-1;Pred(pred1>=prah)=1;
    ACC1(i) = sum( sum( (GT == Pred) ,2)==3) ./ size(Pred,1) *100;
    PRAH1(i) = prah;
    i=i+1;
end
RMSE = sqrt( sum(((GT - pred1).^2),'All')/(numel(GT)) )


GT = cell2mat(labels2);
i=1;ACC2=[];PRAH2=[];
for prah = -1:0.01:1
    Pred = pred2;Pred(pred2<prah)=-1;Pred(pred2>=prah)=1;
    ACC2(i) = sum( sum( (GT == Pred) ,2)==3) ./ size(Pred,1) *100;
    PRAH2(i) = prah;
    i=i+1;
end
RMSE = sqrt( sum(((GT - pred2).^2),'All')/(numel(GT)) )

figure
plot(PRAH1,ACC1,'r')
hold on
plot(PRAH2,ACC2,'b')

pred2(pred2>0)=1;
pred2(pred2<0)=-1;
m=0;Labl=zeros(1,size(pred2,1));
for i=[-1,1]
    for ii=[-1,1]
        for iii=[-1,1]
            m = m+1;
            ind5=[];
            vec = [i,ii,iii];
                for k = 1:size(GT,1)
                   ind5(k) = sum(GT(k,:) == vec)==3;
                end
            Labl(logical(ind5)) = m;
        end
    end
end

corect = sum( (GT == pred2) ,2)==3
figure
hist(Labl(corect),8)