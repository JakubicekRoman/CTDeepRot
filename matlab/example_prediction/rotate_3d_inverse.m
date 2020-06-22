function [img] = rotate_3d_inverse(img,angels_deg)

angels_deg=(-angels_deg)/90;
angels_deg=mod(angels_deg,4);

img = rot90_3D(img, 3, angels_deg(3));
img = rot90_3D(img, 2, angels_deg(2));
img = rot90_3D(img, 1, angels_deg(1));

end

