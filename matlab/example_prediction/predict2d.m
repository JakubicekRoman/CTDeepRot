function [fixed_data,angles] = predict2d(data)

rot_table=readtable('rot_dict_unique.csv');
unique_rots=rot_table{:,1:3};

load('models/net2d.mat','net')

[imMean,imMax,imStd]=get_2d_feature_imgs(data);
                
data_2d=cat(3,imMean,imMax,imStd);

data_2d=single(data_2d)/255-0.5;

res=predict(net,data_2d);


[~,ind]=max(res);

angles=unique_rots(ind,:);



fixed_data=rotate_3d_inverse(data,angles);





end

