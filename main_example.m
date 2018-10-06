%% This is an example of DVFA's usage.
%
% PROGRAM DESCRIPTION
% This program exemplifies the usage of the DVFA code provided.
%
% REFERENCES
% [1] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch II, “Dual
% Vigilance Fuzzy ART,” 2018, manuscript submitted for publication.
% [2] L. E. Brito da Silva, D. C. Wunsch II, "A study on exploiting VAT to
% mitigate ordering effects in Fuzzy ART", Proc. Int. Joint Conf. Neural 
% Netw. (IJCNN), 2018, pp. 2351-2358.
% [3] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast 
% stable learning and categorization of analog patterns by an adaptive 
% resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
% [4] J. C. Bezdek and R. J. Hathaway, "VAT: a tool for visual assessment
% of (cluster) tendency", Proc. Int. Joint Conf. Neural Netw. (IJCNN), 
% vol. 3, 2002, pp. 2225-2230.
% 
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clean Run
clear variables; close all; echo off; clc;

%% Add path
addpath('classes', 'functions', 'data');
fprintf('\t\t\t\tRunning Example Code...\n');

%% Load toy data set
fprintf('Loading data...\n');
tic
load('ACIL.mat')
[nSamples, dim] = size(data);
t = toc;
fprintf('\tElapsed Time: %.4f \n', t);

% Linear Normalization
fprintf('Normalizing data...\n');
tic
data = mapminmax(data', 0, 1);
data = data';
data_backup = data;
t = toc;
fprintf('\tElapsed Time: %.4f \n', t);

% Randomize
fprintf('Randomizing data...\n');
tic
seed = 2468; % for reproducibility
generator = 'twister';        
rng(seed, generator); 
Prng = randperm(nSamples);
data_rng = data_backup(Prng, :);
t = toc;
fprintf('\tElapsed Time: %.4f \n', t);

% Reordering using VAT
fprintf('Running VAT...\n');
fprintf('Obs.: If included, this step may take a while depending on the data size.\n');
tic
M = pdist2(data_rng, data_rng);
[R, Pvat] = VAT(M);
data_vat = data_rng(Pvat, :);
t = toc;
fprintf('\tElapsed Time: %.4f \n', t);

%% Dual Vigilance Fuzzy ART
tic
settings = struct();
settings.rho_lb = .93;
settings.rho_ub = .95;
settings.alpha = 1e-3;
settings.beta = 1;
settings.display = true;
nEpochs = 1;
DVFA = DualVigilanceFuzzyART(settings);
DVFA = DVFA.train(data_vat, nEpochs); 
t = toc;
fprintf('\tElapsed Time: %.4f \n', t);
%% Fuzzy ART (uncomment in order to use)
tic
settings = struct();
settings.rho = .95;
settings.alpha = 1e-3;
settings.beta = 1;
nEpochs = 1;
FA = FuzzyART(settings);
FA = FA.train(data_vat, nEpochs); 
t = toc;
fprintf('\tElapsed Time: %.4f \n', t);
%% Plot Categories/Clusters
fprintf('Plotting the clustering solutions...\n');
tic
LineWidth = 2;
figure
subplot(1,2,1)
draw_categories(DVFA, data_vat, LineWidth)
title('DVFA')
subplot(1,2,2)
draw_categories(FA, data_vat, LineWidth)
t = toc;
fprintf('\tElapsed Time: %.4f \n', t);
title('FA')
fprintf('Done.\n');