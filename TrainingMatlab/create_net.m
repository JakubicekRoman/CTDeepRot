%% create net

function lgraph = create_net

lgraph = resnet18;
lgraph = layerGraph(lgraph);

InLayer = imageInputLayer([224 224 9],'Name','data','Normalization','none','DataAugmentation','none');
lgraph = replaceLayer(lgraph,'data',InLayer);

convLayer = convolution2dLayer(7,64,'Name','conv1','NumChannels',9,'Padding',[3 3 3 3],'Stride',[2,2]);
lgraph = replaceLayer(lgraph,'conv1',convLayer);

fc = fullyConnectedLayer(24,'Name','FcEnd');
lgraph = replaceLayer(lgraph,'fc1000',fc);

SFM = mySoftMax('softmax');
lgraph = replaceLayer(lgraph,'prob',SFM);

classLayer = customClassificationLayer('ClassificationLayer_my');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',classLayer);

% disconnectLayers(lgraph,'');

% NewLayersEnd = [  
%     dropoutLayer(0.5,'Name','dropout')
%     fullyConnectedLayer(24,'Name','FcEnd')
%     mySoftMax('softmax')
%     customRegresionLayer('ClassificationLayer_predictions') ]
% 
% lgraph = [ lgraph layerGraph(NewLayersEnd) ];

