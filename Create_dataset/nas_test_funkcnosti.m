clear all

path_save_img4 = 'test\';
mkdir(path_save_img4)

path_raw = ['\\nas1.ubmi.feec.vutbr.cz\Data\PHILIPS\PHILIPS\MELDOLA\DATA\DIR\raw\' 'DIR_Data_3.mhd' ];
data = load_raw(path_raw);

data = imresize3(data,[201,201,201]);

num = 1;
    for rotX = 0:90:270
        data1 = imrotate3(data,rotX,[1,0,0],'nearest','crop');
        for rotY = 0:90:270
            data2 = imrotate3(data1,rotY,[0,1,0],'nearest','crop'); % rotY - rotace podle X osy
            for rotZ = 0:90:270
                data3 = imrotate3(data2,rotZ,[0,0,1],'nearest','crop'); % rotY - rotace podle X osy

                lbl(num,1) = rotX;
                lbl(num,2) = rotY;
                lbl(num,3) = rotZ;
                v = codingAngle([rotX,rotY,rotZ]);
                lbl(num,4:6) = v;
            
            NUM = ['000' num2str(num)]; NUM = NUM(end-2:end);
                
             prah1 = 100; prah2 = 3200;
                    img = single(squeeze(max(data3,[],3)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    IM(:,:,1,num) = uint8((img).*255);
                    
                    img = single(squeeze(max(data3,[],2)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                     IM(:,:,2,num) = uint8((img).*255);

                    img = single(squeeze(max(data3,[],1)));
                    img(img>prah2)=0;img(img<prah1)=0;m = prah2-prah1; img = (img - prah1) ./ (m);
                    IM(:,:,3,num) = uint8((img).*255);
 
                num = num+1;
            end
        end
    end
                
    
    for i = 1:64
      for ii = 1:64
           M(i,ii) = sum(IM(:,:,:,i) - IM(:,:,:,ii),'All');
      end
    end
    
%%
f = 21;
ff=14;

figure
subplot(231)
imshow(IM(:,:,1,f))
subplot(232)
imshow(IM(:,:,2,f))
subplot(233)
imshow(IM(:,:,3,f))
subplot(234)
imshow(IM(:,:,1,ff))
subplot(235)
imshow(IM(:,:,2,ff))
subplot(236)
imshow(IM(:,:,3,ff))

t = double(~logical(M>100000)) .* tril(ones(size(M)),-1)
ind = (sum(t,2))==0

sum(ind)

new_lbl = lbl(ind,:)
                