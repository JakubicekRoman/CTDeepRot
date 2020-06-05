%% augmentation
clear all
close all
clc

path_save = ['C:\Data\Jakubicek\CTDeepRot_data\training'];
% path_save = ['C:\Data\Jakubicek\CTDeepRot_data\testing'];
mkdir(path_save)

path_save_img1 = [path_save '\' 'mean_20'];
mkdir(path_save_img1);

path_save_img2 = [path_save '\' 'mean_All'];
mkdir(path_save_img2);

path_save_img3 = [path_save '\' 'max_40'];
mkdir(path_save_img3);

path_save_img4 = [path_save '\' 'max_All'];
mkdir(path_save_img4);

path_save_img5 = [path_save '\' 'std_40'];
mkdir(path_save_img5);

path_save_img6 = [path_save '\' 'std_All'];
mkdir(path_save_img6);

path_data = '\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\VerSe2019\Data_raw_train';
% path_data = '\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\VerSe2019\Data_raw_test';


D = dir([path_data '\*.mhd']);

% lbl = readcell([path_save '\labels.xlsx']);
lbl  = {};

DAT = '02';
numAll = size(lbl,1) + 1;

for pat = [1:size(D,1)]
    Data = load_raw([D(pat).folder '\' D(pat).name]);
%     if size(Data,3)>600
%         Data = imresize3(Data,[size(Data,1),size(Data,2),400]);
%     end
%     if pat == 3; Data = Data(250:end-300,:,:); 
%     elseif pat == 25; Data = Data(1:2:end,1:2:end,:); 
%     end
    
    if size(Data,1)>400 && size(Data,2)>400 && size(Data,3)>600
        Data = imresize3(Data,[300,300,400]);
    end
 
pat
num = 1;
tic
for permut = 0:1   
  if permut==1;data1 = flip(Data,2); else data1=Data; end
    for rotX = 0:90:270
       data2 = imrotate3(data1,rotX,[1,0,0],'nearest');
        for rotY = 0:90:270
          data3 = imrotate3(data2,rotY,[0,1,0],'nearest'); % rotY - rotace podle X osy
            for rotZ = 0:90:270
               data = imrotate3(data3,rotZ,[0,0,1],'nearest'); % rotZ - rotace AXIAL                         
                    
                    PAT = ['000' num2str(pat)]; PAT = PAT(end-2:end);
                    NUM = ['000' num2str(num)]; NUM = NUM(end-2:end);
                    name = ['D' DAT '_P' PAT '_A' NUM];

                %% mean 21
                p=10;
                prah1 = 500; prah2 = 2200;
                    img = squeeze(mean(single(data(:,:,ceil(size(data,3)./2)-p:ceil(size(data,3)./2)+p)),3));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img1 '\' name '_R1_Ch1' '.png'] )
                    
                    img = squeeze(mean(single(data(:,ceil(size(data,2)./2)-p:ceil(size(data,2)./2)+p,:)),2));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img1 '\' name '_R1_Ch3' '.png'] )
                    
                    img = squeeze(mean(single(data(ceil(size(data,1)./2)-p:ceil(size(data,1)./2)+p,:,:)),1));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img1 '\' name '_R1_Ch2' '.png'] )
                    
                %% mean All    
                prah1 = 50; prah2 = 1800;
                    img = squeeze(mean(single(data),3));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img2 '\' name '_R2_Ch1' '.png'] )
                    
                    img = squeeze(mean(single(data),2));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img2 '\' name '_R2_Ch3' '.png'] )
                    
                    img = squeeze(mean(single(data),1));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img2 '\' name '_R2_Ch2' '.png'] )
                    
                %% max 41
                p = 20; if sum(size(data)<=(p*2+1)); p = floor(min((size(data)-3)/2));end
                prah1 = 0; prah2 = 3000;
%                     img = single(squeeze(max(data,[],3)));
                    img = single(squeeze(max(data(:,:,ceil(size(data,3)./2)-p:ceil(size(data,3)./2)+p),[],3)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img3 '\' name '_R3_Ch1' '.png'] )
                    
%                     img = single(squeeze(max(data,[],2)));
                    img = single(squeeze(max(data(:,ceil(size(data,2)./2)-p:ceil(size(data,2)./2)+p,:),[],2)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img3 '\' name '_R3_Ch3' '.png'] )
                    
%                     img = single(squeeze(max(data,[],1)));
                    img = single(squeeze(max(data(ceil(size(data,1)./2)-p:ceil(size(data,1)./2)+p,:,:),[],1)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img3 '\' name '_R3_Ch2' '.png'] )
                    
                %% max All
                prah1 = 100; prah2 = 3200;
                    img = single(squeeze(max(data,[],3)));
%                     img = single(squeeze(max(data(:,:,ceil(size(data,3)./2)-p:ceil(size(data,3)./2)+p),[],3)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img4 '\' name '_R4_Ch1' '.png'] )
                    
                    img = single(squeeze(max(data,[],2)));
%                     img = single(squeeze(max(data(:,ceil(size(data,2)./2)-p:ceil(size(data,2)./2)+p,:),[],2)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img4 '\' name '_R4_Ch3' '.png'] )
                    
                    img = single(squeeze(max(data,[],1)));
%                     img = single(squeeze(max(data(ceil(size(data,2)./2)-p:ceil(size(data,2)./2)+p,:,:),[],1)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img4 '\' name '_R4_Ch2' '.png'] )
                    
                %% std 40
                p = 20; if sum(size(data)<=(p*2+1)); p = floor(min((size(data)-3)/2));end
                prah1 = 0; prah2 = 1000;
                    img = squeeze(std(single(data(:,:,ceil(size(data,3)./2)-p:ceil(size(data,3)./2)+p)),0,3));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img5 '\' name '_R5_Ch1' '.png'] )
                    
                    img = squeeze(std(single(data(:,ceil(size(data,2)./2)-p:ceil(size(data,2)./2)+p,:)),0,2));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img5 '\' name '_R5_Ch3' '.png'] )
                    
                    img = squeeze(std(single(data(ceil(size(data,1)./2)-p:ceil(size(data,1)./2)+p,:,:)),0,1));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img5 '\' name '_R5_Ch2' '.png'] )
                    
                %% std All
                prah1 = 0; prah2 = 1000;
                    img = squeeze(std(single(data),0,3));
%                     img = single(squeeze(max(data(:,:,ceil(size(data,3)./2)-p:ceil(size(data,3)./2)+p),[],3)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img6 '\' name '_R6_Ch1' '.png'] )
                    
                    img = squeeze(std(single(data),0,2));
%                     img = single(squeeze(max(data(:,ceil(size(data,2)./2)-p:ceil(size(data,2)./2)+p,:),[],2)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img6 '\' name '_R6_Ch3' '.png'] )
                    
                    img = squeeze(std(single(data),0,1));
%                     img = single(squeeze(max(data(ceil(size(data,2)./2)-p:ceil(size(data,2)./2)+p,:,:),[],1)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    imwrite(uint8((img).*255),[path_save_img6 '\' name '_R6_Ch2' '.png'] )

                %% ground truth
                    lbl{numAll,1} = name;
                    lbl{numAll,2} = rotX;
                    lbl{numAll,3} = rotY;
                    lbl{numAll,4} = rotZ;
                    lbl{numAll,5} = permut;

                    num = num+1;
                    numAll=numAll+1;
                end
            end
        end
    end
    num = 1;   
    toc
end

writecell(lbl,[path_save '\labels.xlsx'])
