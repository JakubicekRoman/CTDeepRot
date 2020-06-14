function [GT] = ReaderValid_class(filename)

G = load(filename);
GT = zeros(24,1);
GT(G.GT) = 1;

% GT = categorical(GT);