function [data] = ReadData3D_lbl(name)

file_name=name(1:end-20);
rotations=name(end-18:end-4);
x_rot=str2num(rotations(1:3));
y_rot=str2num(rotations(5:7));
z_rot=str2num(rotations(9:11));
flip=str2num(rotations(13:15));


data = codingAngle([x_rot,y_rot,z_rot]);



end

