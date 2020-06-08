function im = squareCropResize(im,outputSize)
            
     
            a = min([size(im,1),size(im,2)]);
            x = (size(im,2) - a) / 2;
            y = (size(im,1) - a) / 2;       
            
%             im = cropIm(im,[x,y,a,a]);
            
            ippResizeSupportedWithCast = isa(im,'int8') || isa(im,'uint16') || isa(im,'int16');
            ippResizeSupportedForType = isa(im,'uint8') || isa(im,'single');
            ippResizeSupported = ippResizeSupportedWithCast || ippResizeSupportedForType;
            
            if ippResizeSupportedWithCast
                im = single(im);
            end
            
            if ippResizeSupported
                im = nnet.internal.cnnhost.resizeImage2D(im,outputSize,'linear',true);
            else
                im = imresize(im,'OutputSize',outputSize,'method','bilinear');
            end
            
            
end
        
     
function B = cropIm(A,rect)
            % rect is [x y width height] in floating point.
            % Convert from (x,y) real coordinates to [m,n] indices.
            rect = floor(rect);
            
            m1 = rect(2) + 1;
            m2 = rect(2) + rect(4);
            
            n1 = rect(1) + 1;
            n2 = rect(1) + rect(3);
                        
            m1 = min(size(A,1),max(1,m1));
            m2 = min(size(A,1),max(1,m2));
            n1 = min(size(A,2),max(1,n1));
            n2 = min(size(A,2),max(1,n2));
            
            B = A(m1:m2, n1:n2, :, :);
end