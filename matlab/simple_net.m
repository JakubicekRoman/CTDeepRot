function lgraph = simple_net()

input_size=128;
lvls=5;
lvl1_filters=4;
convs_in_layer=3;
output_size=24;

layers=[];

layer = image3dInputLayer([input_size,input_size,input_size],'Normalization','none','Name','input');
layers=[layers,layer];


for lvl=1:lvls
    for conv_num_in_layer=1:convs_in_layer
        layer = convolution3dLayer(3,lvl1_filters*lvl,'Padding','same','Name',['c' num2str(conv_num_in_layer) '_lvl' num2str(lvl)]);
        layers=[layers,layer];
        layer = reluLayer('Name',['r' num2str(conv_num_in_layer) '_lvl' num2str(lvl)]);
        layers=[layers,layer];
        layer = batchNormalizationLayer('Name',['bn' num2str(conv_num_in_layer) '_lvl' num2str(lvl)]);
        layers=[layers,layer];
    end
    
    
    layer = additionLayer(2,'Name',['add_lvl' num2str(lvl)]);
    layers=[layers,layer];
    
    layer = maxPooling3dLayer(2,'stride',2,'Padding','same','Name',['pool_lvl' num2str(lvl)]);
    layers=[layers,layer];

end

layer = fullyConnectedLayer(output_size,'Name','fc');
layers=[layers,layer];
% layer = regressionLayer('Name','routput');
layer =mySoftMax('sm');
layers=[layers,layer];

layer =customClassificationLayer('coutput');
layers=[layers,layer];



lgraph=layerGraph(layers);

for lvl=1:lvls
    lgraph=connectLayers(lgraph,['c' num2str(1) '_lvl' num2str(lvl)],['add_lvl' num2str(lvl) '/in2']);
    
end

