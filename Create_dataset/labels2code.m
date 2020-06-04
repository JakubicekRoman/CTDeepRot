clear all
close all
clc

path_data = ['\\nas1.ubmi.feec.vutbr.cz\Data\CELL\sdileni_jirina_roman_tom\CT_rotation_data_sample\training'];

lbl = readcell([path_data '\labels.xlsx']);

Angle = cell2mat(lbl(:,2:4));

Angle = Angle./90;
Angles = cat(2,  de2bi(Angle(:,1)) , de2bi(Angle(:,2)) , de2bi(Angle(:,3)) , cell2mat(lbl(:,5)) );

lbl2 = lbl(:,1);
lbl2(:,2:8) = mat2cell(Angles,ones(size(Angles,1),1),ones(size(Angles,2),1));

writecell(lbl2,[path_data '\labels_bin.xlsx']);

