function [GT] = ReaderValid(filename)

GT = load(filename);
GT = GT.GT;
GT = codingAngle(GT(1:3))' ;