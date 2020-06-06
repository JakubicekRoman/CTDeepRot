function [GT] = ReaderValid_class(filename)

GT = load(filename);
GT = GT.GT;
GT = GT(1:2);
if GT(1)<0 | GT(2)<0
    GT = acosd(abs(GT(1:2))* [0;1]) +180;
else
    GT = acosd(abs(GT(1:2))* [0;1]);
end
GT = categorical(GT);