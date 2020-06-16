function lgraph=resnet_2d()

lgraph = layerGraph(resnet18('Weights','imagenet'));

input_size=224;
output_size=24;
features=9;

layer= imageInputLayer([input_size,input_size,features],'Normalization','none','Name','input');
lgraph = replaceLayer(lgraph,'data',layer);

layer = convolution2dLayer(7,64,'Stride',[2 2],'Padding','same','Name','conv1');
lgraph = replaceLayer(lgraph,'conv1',layer);

layer = fullyConnectedLayer(output_size,'Name','fc');
lgraph = replaceLayer(lgraph,'fc1000',layer);



layer =mySoftMax('sm');
lgraph = replaceLayer(lgraph,'prob',layer);

layer =customClassificationLayer('coutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',layer);


end