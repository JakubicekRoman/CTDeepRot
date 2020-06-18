classdef mySoftMax  < nnet.layer.Layer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
    end

    methods
        function layer = mySoftMax(name)           
            % (Optional) Create a myRegressionLayer.

            % Layer constructor function goes here.
            layer.Name = name;
        end
        
        
        function Z = predict(layer,X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            

            
%             Z =  1 ./ (1 + exp(-X));
%             X = squeeze(X);
            D=find(size(X)==24);
            if isempty(D)
                error('different nuber of outputs')
            end

            b=max(X,[],D);
            
            
            X=X-b;

            Z =  exp(X) ./ sum(exp(X),D);
            

            
            
        end
        
        function dLdX = backward(layer, ~, Z, dLdZ, ~)
            % Backward propagate the derivative of the loss function through 
            % the layer
            
            
            
            d = Z.*(1 - Z);
            dLdX = d .* dLdZ;
        end
    end
end