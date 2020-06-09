%% validace nets
clear all;close all force;clc


% load('Trained_nets\Net_1_2.mat')
load('Trained_nets\Net_1_regr_8.mat')

%% datastore



imdsValid1 = imdsTrainComb.UnderlyingDatastores{1};
labels1 = imdsTrainComb.UnderlyingDatastores{2}.Files;
labels1 = cellfun(@(x) ReadData3D_lbl(x), labels1,'UniformOutput',false);


imdsValid2 = imdsTestComb.UnderlyingDatastores{1};
labels2 = imdsTestComb.UnderlyingDatastores{2}.Files;
labels2 = cellfun(@(x) ReadData3D_lbl(x), labels2,'UniformOutput',false);

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
