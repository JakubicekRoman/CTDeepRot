classdef customRegresionLayer < nnet.layer.RegressionLayer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
    end

    methods
        function layer = customRegresionLayer(name)           
            % (Optional) Create a myRegressionLayer.

            % Layer constructor function goes here.
            layer.Name = name;
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here.
            
%             q=0.5*(Y-T).^2;
%             loss=sum(q(:))/numel(q);   
            Y=squeeze(Y);
            T=squeeze(T);

            Y=(Y>0);
            T=(T>0);
            
            loss=single(mean(sum(Y==T,1)==size(Y,1)));
            
            
            drawnow;
            
            
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Backward propagate the derivative of the loss function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y        

            % Layer backward loss function goes here.
            
            q=Y-T;
%             q=sign(Y-T);
            
            
            dLdY=q/numel(q);
            
        end
    end
end