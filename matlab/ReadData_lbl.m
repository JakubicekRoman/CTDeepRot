function [data] = ReadData_lbl(name)

file_name=name(1:end-20);
rotations=name(end-18:end-4);
x_rot=str2num(rotations(1:3));
y_rot=str2num(rotations(5:7));
z_rot=str2num(rotations(9:11));
flip=str2num(rotations(13:15));



unique_rots=[180,0,0;180,0,90;180,0,180;180,0,270;180,180,0;180,180,90;180,180,180;180,180,270;270,0,0;270,0,90;270,0,180;270,0,270;270,90,0;270,90,90;270,90,180;270,90,270;270,180,0;270,180,90;270,180,180;270,180,270;270,270,0;270,270,90;270,270,180;270,270,270];
unique_rots=num2cell(unique_rots,2)';
data = single(cellfun(@(x) all(x==[x_rot,y_rot,z_rot]),unique_rots));



end
