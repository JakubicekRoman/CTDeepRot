function [angle] = codingAngle(x)

Rz =    [cosd(x(3)),-sind(x(3)),0,0;...
        sind(x(3)),cosd(x(3)),0,0;...
        0,0,1,0;...
        0,0,0,1];
Ry = [1, cosd(x(2)),-sind(x(2)),0;...
    0,1,0,0;...
    0, sind(x(2)),cosd(x(2)),0;...
    0,0,0,1];

Rx = [1,0,0,0;...
    0, cosd(x(1)),-sind(x(1)),0;...
    0, sind(x(1)),cosd(x(1)),0;...
    0,0,0,1];

R = Rx * Ry * Rz;

angle = R * [1,1,1,1]';
angle = angle(1:3);

