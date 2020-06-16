function img=flip_2d(img)
    

img(:,:,1)=flipud(img(:,:,1));
img(:,:,2)=fliplr(img(:,:,2));
