function [GT] = ReaderValid_class(filename)

GT = load(filename);
GT = GT.GT;
GT = GT(1:2);

GT = categorical(GT);