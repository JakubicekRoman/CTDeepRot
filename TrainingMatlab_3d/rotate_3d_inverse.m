function [img] = rotate_3d_inverse(img,angels_deg)

img = rot90_3D(img, 3, angels_deg(3)/90);
img = rot90_3D(img, 2, angels_deg(2)/90);
img = rot90_3D(img, 1, angels_deg(1)/90);

end

