function [GT] = ReaderValid(filename)

GT = load(filename);
GT = GT.GT;
GT = GT(1:6);