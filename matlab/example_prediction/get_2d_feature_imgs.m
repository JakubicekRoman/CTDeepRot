function [imMean,imMax,imStd]=get_2d_feature_imgs(data)
    
    size=224;
    

    t1 = 50; t2 = 1800;
    imMean=zeros(size,size,3,'uint8');
    for dim=1:3
        img = squeeze(mean(single(data),dim));
        img=mat2gray(img,[t1,t2]);
        img=imresize(img,[size,size]);
        img= uint8(img*255);
        imMean(:,:,dim)=img;
    end
    
    
    t1 = 100; t2 = 3200;
    imMax=zeros(size,size,3,'uint8');
    for dim=1:3
        img = squeeze(max(single(data),[],dim));
        img=mat2gray(img,[t1,t2]);
        img=imresize(img,[size,size]);
        img= uint8(img*255);
        imMax(:,:,dim)=img;
    end
    
    
    t1 = 0; t2 = 1000;
    imStd=zeros(size,size,3,'uint8');
    for dim=1:3
        img = squeeze(std(single(data),[],dim));
        img=mat2gray(img,[t1,t2]);
        img=imresize(img,[size,size]);
        img= uint8(img*255);
        imStd(:,:,dim)=img;
    end
    
    
end