

function [rot,vec] = get_rot_vec_pairs()

vec={};
rot={};

unique_rots={};
for a=[0,90,180,270]
    for b=[0,90,180,270]
        for c=[0,90,180,270]

            rot_vec = codingAngle([a,b,c]);

            new=sum(cellfun(@(x) all(x==rot_vec),unique_rots))==0;

            if new
                vec=[vec,rot_vec];
                rot=[rot,[a,b,c]];
                
                unique_rots=[unique_rots,rot_vec];
                if length(unique_rots)==8
                    return
                end
            end

            
        end
    end
end

end


