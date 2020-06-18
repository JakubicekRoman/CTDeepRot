

clc;clear all;close all;

theta=30/180*pi;
sx=1.2;
sy=0.7;

tx=10;
ty=10;


S=[sx 0 0; 0 sy 0; 0 0 1];
R=[cos(theta) -sin(theta) tx;sin(theta) cos(theta) tx; 0 0 1];
T=S*R;


theta_rec=atan2(-T(1,2),T(1,1));


sx=T(1,1)/cos(theta_rec);
sy=T(1,1)/sin(theta_rec);

tx=T(1,3)/sx;

ty=T(2,3)/sy;


