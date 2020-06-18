  
classdef customClassificationLayer < nnet.layer.RegressionLayer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
    end

    methods
        function layer = customClassificationLayer(name)           
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

        [~,Y]=max(Y,[],1);
        [~,T]=max(T,[],1);

        loss=single(mean(Y==T));
      
    
            
            
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
            
%             q=Y-T;
%             q=sign(Y-T);

            Y(Y< eps('single')) = eps('single');

            q = -(T./Y);
            
            
            
            
            dLdY = (q)/numel(q);
            
            tmp=isnan(dLdY);
            s=sum(tmp(:));
            if s>0
                disp(['NaNs in dL  '  num2str(s)])
                dLdY(tmp)=0;
            end
            
            
        end
    end
end