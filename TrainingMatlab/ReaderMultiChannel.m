function [im] = ReaderMultiChannel(filename)

% folders = {'max_40','max_All','mean_20','mean_All','std_40','std_All'};
folders = {'max_40','std_All'};
R = {'R3','R6'};
<<<<<<< HEAD
=======
% R = {'R3','R4','R1','R2','R5','R6'};
>>>>>>> 99be23a5d877ff86e57bde57463ac74726901c7e
CH = {'Ch1','Ch2','Ch3'};

i=1;
for f = 1:length(folders)
    for ch = 1:3
        filename = strrep(filename,'mean_20',folders{f});
        filename = strrep(filename,'R1',R{f});
        filename = strrep(filename,'Ch1',CH{ch});
        
        img = im2double(imread(filename))-0.5;
        img = imresize(img,[224,224]);
        im(:,:,i) = img;
        i=i+1;
    end
end