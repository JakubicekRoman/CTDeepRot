clc;clear all;close all;


permss={};

rots_2d={};

rots_3d={};


s1=10;


matice=rand(10,10,10);

X=zeros(s1,s1,s1,64);
i=0;
for a=[0,90,180,270]
    for b=[0,90,180,270]
        for c=[0,90,180,270]
            i=i+1;
            data = rot90_3D(matice, 1, a/90);
            data = rot90_3D(data, 2, b/90);
            data = rot90_3D(data, 3, c/90);
            
            X(:,:,:,i)=data;
            
            
            for p=perms(1:3)'
                for rot1=0:3
                    for rot2=0:3
                        for rot3=0:3
                            tmp=(data-matice).^2;
                            d=sum(tmp(:));
                            if d==0
                                permss=[permss,p];

                                rots_2d=[rots_2d,[rot1,rot2,rot3]];
                                
                                
                                rots_3d=[rots_3d,[a,b,c]];

                            end
                        end
                        
                    end
                    
                end
                
            end
        end
    end
end



v_d=reshape(X,s1*s1*s1,64);

D = pdist(v_d');

D = squareform(D);

D=tril(D==0,-1);

ind=find(sum(D) ==0);

if length(ind)~=24
    errror('smůla')
end



