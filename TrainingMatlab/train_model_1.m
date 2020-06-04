clear all
close all
clc

lgraph = resnet50();

lgraph = layerGraph(lgraph.Layers);
lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});

newLayers = [
    fullyConnectedLayer(7,'Name','fc1000','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = addLayers(lgraph,newLayers);

newLayers = imageInputLayer([224,224,18],'Name','input');
lgraph = replaceLayer(lgraph,'input_1',newLayers);

trainData = readtable('train_dataset_001.xlsx');
testData = readtable('test_dataset_001.xlsx');

options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',16, ...
    'Plots','training-progress');

net = trainNetwork(trainData,lgraph,options);


