function [data] = ReadData2D(name)

MEAN=900.3071;
STD=318.11615;

file_name=name(1:end-20);
rotations=name(end-18:end-4);

x_rot=str2num(rotations(1:3));
y_rot=str2num(rotations(5:7));
z_rot=str2num(rotations(9:11));
flip=str2num(rotations(13:15));



data=zeros(224,224,9);
ind=0;
for folder = {'mean','max','std'}
    for k=1:3
        ind=ind+1;
        tmp=imread([file_name '_' folder{1}  '_'  num2str(k) '.png'] );
        tmp=single(tmp)/255-0.5;
        data(:,:,ind)=tmp;
    end
    
end



if flip
    data(:,:,1:3)=flip_2d(data(:,:,1:3));
    data(:,:,4:6)=flip_2d(data(:,:,4:6));
    data(:,:,7:9)=flip_2d(data(:,:,7:9));
end

data(:,:,1:3) = rotate_2d(data(:,:,1:3),[x_rot,y_rot,z_rot]);
data(:,:,4:6) = rotate_2d(data(:,:,4:6),[x_rot,y_rot,z_rot]);
data(:,:,7:9) = rotate_2d(data(:,:,7:9),[x_rot,y_rot,z_rot]);




end