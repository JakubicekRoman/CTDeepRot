function imAll = create_features(data)

prah1 = 50; prah2 = 1500;
img = squeeze(mean(single(data),3));
img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
imAll(:,:,1) = squareCropResize(im2double(img),[224,224])-0.5;

img = squeeze(mean(single(data),1));
img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
imAll(:,:,2) = squareCropResize(im2double(img),[224,224])-0.5;

img = squeeze(mean(single(data),2));
img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
imAll(:,:,3) = squareCropResize(im2double(img),[224,224])-0.5;



prah1 = 0; prah2 = 1000;
img = squeeze(std(single(data),0,3));
img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
imAll(:,:,4) = squareCropResize(im2double(img),[224,224])-0.5;

img = squeeze(std(single(data),0,1));
img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
imAll(:,:,5) = squareCropResize(im2double(img),[224,224])-0.5;

img = squeeze(std(single(data),0,2));
img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
imAll(:,:,6) = squareCropResize(im2double(img),[224,224])-0.5;


prah1 = 100; prah2 = 3200;
img = single(squeeze(max(data,[],3)));
img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
imAll(:,:,7) =  squareCropResize(im2double(img),[224,224])-0.5;

img = single(squeeze(max(data,[],1)));
img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
imAll(:,:,8) =  squareCropResize(im2double(img),[224,224])-0.5;

img = single(squeeze(max(data,[],2)));
img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
imAll(:,:,9) =  squareCropResize(im2double(img),[224,224])-0.5;



% prah1 = 100; prah2 = 3200;
% img = single(squeeze(max(data,[],3)));
% img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
% imAll(:,:,1) =  squareCropResize(im2double(img),[224,224])-0.5;
% 
% img = single(squeeze(max(data,[],1)));
% img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
% imAll(:,:,2) =  squareCropResize(im2double(img),[224,224])-0.5;
% 
% img = single(squeeze(max(data,[],2)));
% img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
% imAll(:,:,3) =  squareCropResize(im2double(img),[224,224])-0.5;
