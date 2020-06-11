function  angle = vec2angle(vec)


[angles,vecs] = get_rot_vec_pairs();




angle=angles{find(cellfun(@(x) all(vec==x),vecs))};





end

