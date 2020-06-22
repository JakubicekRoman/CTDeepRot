function [fixed_data,angles] = predict3d(data)

data_in=imresize3(data,[128,128,128]);

rot_table=readtable('rot_dict_unique.csv');
unique_rots=rot_table{:,1:3};

load('models/net3d.mat','net')

MEAN=900.3071;
STD=318.11615;

data_in=(data_in-MEAN)/STD;
res=predict(net,data_in);


[~,ind]=max(res);

angles=unique_rots(ind,:);



fixed_data=rotate_3d_inverse(data,angles);





end

