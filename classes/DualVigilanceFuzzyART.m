%% """ Dual Vigilance Fuzzy ART (DVFA)"""
% 
% PROGRAM DESCRIPTION
% This is a MATLAB implementation of the "Dual Vigilance Fuzzy ART (DVFA)" network.
%
% REFERENCES
% [1] L. E. Brito da Silva, I. Elnabarawy and D. C. Wunsch II, "Dual 
% Vigilance Fuzzy ART," Neural Networks Letters. To appear.
% [2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast 
% stable learning and categorization of analog patterns by an adaptive 
% resonance system," Neural Networks, vol. 4, no. 6, pp. 759–771, 1991.
% 
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dual Vigilance Fuzzy ART (DVFA) Class
classdef DualVigilanceFuzzyART    
    properties (Access = public)        % default properties' values are set
        rho_lb = 0.55;                  % lower bound vigilance parameter: [0,1]  
        rho_ub = 0.6;                   % upper bound vigilance parameter: [0,1] 
        alpha = 1e-3;                   % choice parameter: (0, Inf)
        beta = 1;                       % learning parameter: (0,1] (beta=1: "fast learning")        
        W = [];                         % weight vectors 
        map = [];                       % category-to-cluster mapping matrix
        labels = [];                    % class labels
        dim = [];                       % original dimension of data set  
        nCategories = 0;                % total number of categories
        nClusters = 0;                  % total number of clusters
        Epoch = 0;                      % current epoch  
        display = true;                 % displays training progress on the command window (displays intermediate steps)
    end   
    properties (Access = private)
        T = [];                         % category activation/choice function vector
        M = [];                         % category match function vector  
        W_old = [];                     % old weight vectors
    end
    methods        
        %% Assign property values from within the class constructor
        function obj = DualVigilanceFuzzyART(settings) 
            obj.rho_lb = settings.rho_lb;
            obj.rho_ub = settings.rho_ub;
            obj.alpha = settings.alpha;
            obj.beta = settings.beta;
        end         
        %% Train
        function obj = train(obj, data, maxEpochs) 
            
            % Display progress on command window
            if obj.display
                fprintf('Trainnig DVFA...\n');                
                backspace = '';          
            end  
            
            % Data Information            
            [nSamples, obj.dim] = size(data);
            obj.labels = zeros(nSamples, 1);
            
            % Normalization and Complement coding
            x = DualVigilanceFuzzyART.complement_coder(data);

            % Initialization 
            if isempty(obj.W)             
                obj.W = ones(1, 2*obj.dim);                                   
                obj.nCategories = 1;       
                obj.nClusters = 1;                
                obj.map(1, 1) = 1;
            end              
            obj.W_old = obj.W;  
            
            % Learning
            obj.Epoch = 0;
            backspace = '';
            while(true)
                obj.Epoch = obj.Epoch + 1;                
                for i=1:nSamples %loop over samples 
                    if or(isempty(obj.T), isempty(obj.M)) % Check for already computed activation/match values
                        obj = activation_match(obj, x(i,:));  % Compute Activation/Match Functions
                    end 
                    [~, index] = sort(obj.T, 'descend');  % Sort activation function values in descending order                    
                    mismatch_flag = true;  % mismatch flag   
                    for j=1:obj.nCategories % For the number of categories                        
                        bmu = index(j);  % Best Matching Unit 
                        if obj.M(bmu) >= obj.rho_ub*obj.dim % Vigilance test upper bound - pass                              
                            class_assignment = find(obj.map(bmu,:));  % cluster assignment
                            obj = learn(obj, x(i,:), bmu);  % learning
                            obj.labels(i) = class_assignment;  % update sample labels
                            mismatch_flag = false;  % set mismatch flag to false
                            break;
                        elseif obj.M(bmu) >= obj.rho_lb*obj.dim  % Vigilance test lower bound - pass                         
                            class_assignment = find(obj.map(bmu,:));  % cluster assignment
                            obj.map = [obj.map; obj.map(bmu,:)]; % append to the mapping a new category with the same class  
                            obj.W(end+1,:) = x(i,:);  % append new category                                
                            obj.labels(i) = class_assignment; % same label / different category
                            obj.nCategories = obj.nCategories + 1; % increment the number of categories        
                            mismatch_flag = false;  % set mismatch flag to false
                            break;
                        end                               
                    end  
                    if mismatch_flag
                        [nrows, ncolumns] = size(obj.map);
                        obj.map = [obj.map zeros(nrows, 1)];  % new column for new class
                        obj.map = [obj.map; zeros(1, ncolumns+1)]; % new row for new category  
                        obj.map(end, end) = 1;  % set new class to new category                        
                        obj.W(end+1,:) = x(i,:); % append new category    
                        obj.nCategories = obj.nCategories + 1;  % increment the number of categories  
                        obj.nClusters = obj.nClusters + 1;  % increment the number of clusters  
                        class_assignment = find(obj.map(end,:));  % cluster assignment  
                        obj.labels(i) = class_assignment; % sample label                                                                   
                    end  
                    obj.T = [];  % clear activation vector
                    obj.M = [];  % clear match vector
                    % Display progress on command window
                    if obj.display
                       progress = sprintf('\tEpoch: %d \n\tSample ID: %d \n\tCategories: %d \n\tClusters: %d \n', obj.Epoch, i, obj.nCategories, obj.nClusters);
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