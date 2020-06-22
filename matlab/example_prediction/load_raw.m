function [data,info] = load_raw(path)

% input is file path with .mhd

fin=fopen([path]);
name = (strjoin(cellstr(char(fread(fin)))));
[~, ind2] = regexp(name,'D i m S i z e  =  ');
s = strsplit(name(ind2+1:end),'  ');
ind = regexp(s{1},' ');
s{1}(ind)='';
ind = regexp(s{2},' ');
s{2}(ind)='';
ind = regexp(s{3},' ');
s{3}(ind)='';
vel = [str2num(s{1}), str2num(s{2}), str2num(s{3})];
fclose('all');

fin=fopen([ [path(1:end-4)] '.raw']);
data=fread(fin,vel(1)*vel(2)*vel(3),'uint16=>uint16');
data=reshape(data,vel(1),vel(2),vel(3));
data = permute(data,[2,1,3]);
info.size = vel;
fclose('all');

[~, ind2] = regexp(name,'E l e m e n t S p a c i n g  =  ');
s = strsplit(name(ind2+1:end),'  ');
ind = regexp(s{1},' ');
s{1}(ind)='';
ind = regexp(s{2},' ');
s{2}(ind)='';
ind = regexp(s{3},' ');
s{3}(ind)='';
res = [str2num(s{1}), str2num(s{2}), str2num(s{3})];
info.resolution=res;
