function [data] = ReadData3D(name)

MEAN=900.3071;
STD=318.11615;

file_name=name(1:end-20);
rotations=name(end-18:end-4);
x_rot=str2num(rotations(1:3));
y_rot=str2num(rotations(5:7));
z_rot=str2num(rotations(9:11));
flip=str2num(rotations(13:15));


load(file_name)


data = rot90_3D(data, 1, x_rot/90);
data = rot90_3D(data, 2, y_rot/90);
data = rot90_3D(data, 3, z_rot/90);



data=(data-MEAN)/STD;




end

