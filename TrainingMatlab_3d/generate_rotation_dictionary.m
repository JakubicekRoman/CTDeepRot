% function rot_dict=get_rotation_dictionary()
clc;clear all;close all;

permss={};
rots_2d={};
flips_ud={};
flips_lr={};

rots_3d={};


s1=2;

s = rng;
rng(42);
matice=rand(s1,s1,s1);
rng(s);

X=zeros(s1,s1,s1,64);
i=0;
for a=[0,90,180,270]
    a
    for b=[0,90,180,270]
        for c=[0,90,180,270]
            i=i+1;
            data = rot90_3D(matice, 1, a/90);
            data = rot90_3D(data, 2, b/90);
            data = rot90_3D(data, 3, c/90);
            
            X(:,:,:,i)=data;
            
            [permss,rots_2d,rots_3d,flips_ud,flips_lr]=find_2d_tranform(data,matice,permss,rots_2d,rots_3d,flips_ud,flips_lr,a,b,c);
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
if length(rots_3d)~=64
    errror('spatne')
end


%


rot_dict_full=table(rots_3d,permss,flips_lr,rots_2d);




rots_3d=rots_3d(ind);
permss=permss(ind);
rots_2d=rots_2d(ind);
% flips_ud=flips_ud(ind);
flips_lr=flips_lr(ind);

rot_dict_unique=table(rots_3d,permss,flips_lr,rots_2d);

writetable(rot_dict_full,'rot_dict_full.csv')
writetable(rot_dict_full,'rot_dict_unique.csv')



function [permss,rots_2d,rots_3d,flips_ud,flips_lr]=find_2d_tranform(data,matice,permss,rots_2d,rots_3d,flips_ud,flips_lr,a,b,c)


for zrc1=0:1
    for zrc2=0:1
        for zrc3=0:1
            
            for p=perms(1:3)'
                for rot1=0:3
                    for rot2=0:3
                        for rot3=0:3
                            
                            
                            data_2d=cat(3,squeeze(sum(data,1)),squeeze(sum(data,2)),squeeze(sum(data,3)));
                            
                            matice_2d=cat(3,squeeze(sum(matice,1)),squeeze(sum(matice,2)),squeeze(sum(matice,3)));
                            
                            matice_2d=matice_2d(:,:,p);
                            
                            if zrc1==1
                                matice_2d(:,:,1)=fliplr(matice_2d(:,:,1));
                            end
                            if zrc2==1
                                matice_2d(:,:,2)=fliplr(matice_2d(:,:,2));
                            end
                            if zrc3==1
                                matice_2d(:,:,3)=fliplr(matice_2d(:,:,3));
                            end
                            
                            
                            matice_2d(:,:,1)=rot90(matice_2d(:,:,1),rot1);
                            matice_2d(:,:,2)=rot90(matice_2d(:,:,2),rot2);
                            matice_2d(:,:,3)=rot90(matice_2d(:,:,3),rot3);
                            
                            
                            
                            
                            tmp=(data_2d-matice_2d).^2;
                            d=sum(tmp(:));
                            
                            if d==0
                                permss=[permss;p'];
                                
                                rots_2d=[rots_2d;[rot1,rot2,rot3]];
                                
                                
                                rots_3d=[rots_3d;[a,b,c]];
                                
                                flips_lr=[flips_lr;[zrc1,zrc2,zrc3]];
                                
                                return
                                
                                
                            end
                        end
                    end
                    
                end
            end
            
        end
        
    end
    
end

end