function [im] = ReaderMultiChannel(filename)

% folders = {'max_40','max_All','mean_20','mean_All','std_40','std_All'};
folders = {'max_40','std_All'};
R = {'R3','R6'};
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