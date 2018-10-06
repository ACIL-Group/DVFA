%% """ 2D Visualization of Fuzzy ART hyperboxes """
% 
% PROGRAM DESCRIPTION
% This program plots Fuzzy ART categories (hyperboxes) and data samples 
% according to the partition obtained.
%
% INPUT
% ART: ART class object
% data: data set matrix (rows: samples, columns: features)
% lw: line width of hyperboxes
%
% REFERENCES
% [1] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast 
% stable learning and categorization of analog patterns by an adaptive 
% resonance system," Neural networks, vol. 4, no. 6, pp. 759–771, 1991.
%
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot Categories
function draw_categories(ART, data, lw)
    ClassName = class(ART);
    switch ClassName
        case 'DualVigilanceFuzzyART'  
            clrs = rand(ART.nClusters, 3);
        case 'FuzzyART'
            clrs = rand(ART.nCategories , 3);
    end
    
    [~, dim] = size(data);

    if dim==2     
        gscatter(data(:,1), data(:,2), ART.labels, clrs, '.', 15, 'off')  
        for j=1:ART.nCategories  
            x = ART.W(j, 1);
            y = ART.W(j, 2);
            w = 1 - ART.W(j, 3) - ART.W(j, 1);
            h = 1 - ART.W(j, 4) - ART.W(j, 2);        
            if and((w>0), (h>0))
                pos = [x y w h]; 
                r = rectangle('Position', pos);
                r.FaceColor = 'none';
                switch ClassName
                    case 'DualVigilanceFuzzyART'  
                        r.EdgeColor = clrs(ART.map(j,:)==1, :);
                    case 'FuzzyART'
                        r.EdgeColor = clrs(j,:);
                end
                r.LineWidth = lw;
                r.LineStyle = '-';
                r.Curvature = [0 0]; 
            else
                X = [ART.W(j, 1) 1 - ART.W(j, 3)];
                Y = [ART.W(j, 2) 1 - ART.W(j, 4)];
                l = line(X, Y);
                switch ClassName
                    case 'DualVigilanceFuzzyART'  
                        l.Color = clrs(ART.map(j,:)==1, :);
                    case 'FuzzyART'
                        l.Color = clrs(j,:);
                end
                l.LineStyle = '-';
                l.LineWidth = lw;
                l.Marker = 'none';
            end  
        end
        
        axis square
        box on
        
    end
    
end