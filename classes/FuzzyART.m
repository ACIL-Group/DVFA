%% """ Fuzzy ART (FA)"""
% 
% PROGRAM DESCRIPTION
% This is a MATLAB implementation of the "Fuzzy ART (FA)" network.
%
% REFERENCES
% [1] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast 
% stable learning and categorization of analog patterns by an adaptive 
% resonance system," Neural networks, vol. 4, no. 6, pp. 759–771, 1991.
% 
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fuzzy ART (FA) Class
classdef FuzzyART    
    properties (Access = public)        % default properties' values are set
        rho;                            % vigilance parameter: [0,1] 
        alpha = 1e-3;                   % choice parameter 
        beta = 1;                       % learning parameter: (0,1] (beta=1: "fast learning")    
        W = [];                         % weight vectors
        labels = [];                    % class labels
        dim = [];                       % original dimension of data set  
        nCategories = 0;                % total number of categories
        Epoch = 0;                      % current epoch    
        display = true;                 % displays training progress on the command window (displays intermediate steps)
    end 
    properties (Access = private)
        T = [];                         % category activation/choice function vector
        M = [];                         % category match function vector  
        W_old = [];                     % old weight vectors
    end
    methods        
        % Assign property values from within the class constructor
        function obj = FuzzyART(settings) 
            obj.rho = settings.rho;
            obj.alpha = settings.alpha;
            obj.beta = settings.beta;
        end         
        %% Train
        function obj = train(obj, data, maxEpochs) 
            
            % Display progress on command window
            if obj.display
                fprintf('Trainnig FA...\n');                
                backspace = '';          
            end            
            
            % Data Information            
            [nSamples, obj.dim] = size(data);
            obj.labels = zeros(nSamples, 1);
            
            % Normalization and Complement coding
            x = FuzzyART.complement_coder(data);
                        
            % Initialization 
            if isempty(obj.W)             
                obj.W = ones(1, 2*obj.dim);                                   
                obj.nCategories = 1;                 
            end              
            obj.W_old = obj.W;    
            
            % Learning            
            obj.Epoch = 0;
            backspace = '';
            while(true)
                obj.Epoch = obj.Epoch + 1;
                for i=1:nSamples  % loop over samples
                    if or(isempty(obj.T), isempty(obj.M)) % Check for already computed activation/match values
                        obj = activation_match(obj, x(i,:));  % Compute Activation/Match Functions
                    end     
                    [~, index] = sort(obj.T, 'descend');  % Sort activation function values in descending order                    
                    mismatch_flag = true;  % mismatch flag 
                    for j=1:obj.nCategories  % loop over categories                       
                        bmu = index(j);  % Best Matching Unit 
                        if obj.M(bmu) >= obj.rho*obj.dim % Vigilance Check - Pass 
                            obj = learn(obj, x(i,:), bmu);  % learning
                            obj.labels(i) = bmu;  % update sample labels
                            mismatch_flag = false;  % mismatch flag 
                            break; 
                        end                               
                    end  
                    if mismatch_flag  % If there was no resonance at all then create new category
                        obj.nCategories = obj.nCategories + 1;  % increment number of categories
                        obj.W(obj.nCategories,:) = x(i,:);  % fast commit                         
                        obj.labels(i) = obj.nCategories;  % update sample labels 
                    end 
                    obj.T = [];  % empty activation vector
                    obj.M = [];  % empty match vector
                    % Display progress on command window
                    if obj.display
                       progress = sprintf('\tEpoch: %d \n\tSample ID: %d \n\tCategories: %d \n', obj.Epoch, i, obj.nCategories);
                       fprintf([backspace, progress]);
                       backspace = repmat(sprintf('\b'), 1, length(progress)); 
                    end 
                end                  
                % Stopping Conditions
                if stopping_conditions(obj, maxEpochs)
                    break;
                end 
                obj.W_old = obj.W;                                
            end 
            % Display progress on command window
            if obj.display
                fprintf('Done.\n');
            end  
        end
        %% Activation/Match Functions
        function obj = activation_match(obj, x)              
            obj.T = zeros(obj.nCategories, 1);     
            obj.M = zeros(obj.nCategories, 1); 
            for j=1:obj.nCategories 
                numerator = norm(min(x, obj.W(j, :)), 1);
                obj.T(j, 1) = numerator/(obj.alpha + norm(obj.W(j, :), 1));
                obj.M(j, 1) = numerator;
            end
        end  
        %% Learning
        function obj = learn(obj, x, index)
            obj.W(index,:) = obj.beta*(min(x, obj.W(index,:))) + (1-obj.beta)*obj.W(index,:);                
        end      
        %% Stopping Criteria
        function stop = stopping_conditions(obj, maxEpochs)
            stop = false; 
            if isequal(obj.W, obj.W_old)
                stop = true;                                         
            elseif obj.Epoch >= maxEpochs
                stop = true;
            end 
        end    
    end    
    methods(Static)
        %% Linear Normalization and Complement Coding
        function x = complement_coder(data)
            x = mapminmax(data', 0, 1);
            x = x';
            x = [x 1-x];
        end         
    end
end