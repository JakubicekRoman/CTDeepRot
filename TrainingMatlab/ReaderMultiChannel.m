function [im] = ReaderMultiChannel(filename)

% folders = {'max_40','max_All','mean_20','mean_All','std_40','std_All'};
% folders = {'mean_All','std_All','max_All'};
 folders = {'max_All'};
R = {'R2','R6','R4'};
R = {'R4'};
% R = {'R3','R4','R1','R2','R5','R6'};
CH = {'Ch1','Ch2','Ch3'};

i=1;
for f = 1:length(folders)
    for ch = 1:3
        filenameX = strrep(filename,'max_All',folders{f});
        filenameX = strrep(filenameX,'R4',R{f});
        filenameX = strrep(filenameX,'Ch1',CH{ch});
        
        img = im2double(imread(filenameX))-0.5;
        img = squareCropResize(img,[224,224]);
        im(:,:,i) = img;
        i=i+1;
    end
end