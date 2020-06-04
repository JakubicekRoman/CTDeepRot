function [GT] = ReaderValid(filename)

GT = load(filename);
GT = GT.GT;