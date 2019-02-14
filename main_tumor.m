%  Thanh Nguyen-Duc
%  Ulsan National Institute of Science and Technology
%  thanhnguyen.cse@gmail.com
%
%  Septembet 2018

clear all; close all; clc;
addpath(genpath('./'));

%% Choose a GPU device (we used Titan X with 12GB memory)
gpu_id = 4;
g = gpuDevice(gpu_id);
reset(g);

%% Run GA for searching parameters
runGA = 0;

%%  Options Dictionary
opt.numAtoms = 27; % It must be divided by 3 
opt.numTypes = 3; % We only support 3 sizes of filters

% Atom 1 size
opt.atomSize1X = 15;
opt.atomSize1Y = 15;
opt.atomSize1Z = 20;

% Atom 2 size
opt.atomSize2X = 20;
opt.atomSize2Y = 20;
opt.atomSize2Z = 25;

% Atom 3 size
opt.atomSize3X = 25;
opt.atomSize3Y = 25;
opt.atomSize3Z = 30;

%% Load TV matrices
file_TV = './TV_matrix/tumor/TV.mat';
load(file_TV);
opt.D = D;
opt.Dt = Dt;

%% Search parameter using Genetic Algorithm (require installed Matlab optimization toolbox)
if runGA
    % Load mask
    file_mask = './data/db_tumor/mask/mask_tumor_125.mat';
    load(file_mask);
    mask = mask;

    % Load 3D MRI
    file_name = './data/db_tumor/data/tumor_2.mat';
    load(file_name);
    full = scale_img(full);

    % Undersample using mask
    undersample = undersample_func(full, mask);

    % options 
    opt.isDisplay = 0; % show figures
    opt.isConsole = 1; % show outputs on console

    % options data size
    [x, y, z] = size(undersample);
    opt.dataSizeX = x; 
    opt.dataSizeY = y;
    opt.dataSizeZ = z;

    % Plot fullsample, undersample, mask
    figure(1);
    subplot(1,3,1);
    imshow(full(:,:,1),[0 1]);
    title('Fullsample');

    temp = ifft2(undersample);
    subplot(1,3,2);
    imshow(temp(:,:,1),[0 1]);
    title('Undersample');

    subplot(1,3,3);
    imshow(mask(:,:,1),[0 1]);
    title('Mask');
    drawnow();

    lb = [0.001 0.001  0.001   0.001    1      1     0.001  1   1];
    ub = [ 5     5      5       5       100    100    1     5   5];
    %    alpha  gamma  lambda1 lambda2  rho   sigma  theta  D0  n
    generation = 6;
    population = 200;
    num_params = 9;
    opt.num_iters = 100; % number of iterations
    opt.saveIntermediate = 1;

    fprintf('Genetic Algorith is running, it will take a long time!\n');
    ga = main_GA(full, mask, undersample, generation, population, num_params, lb, ub, opt);
    
    % File save name
    file_save = './searched_params/GA_param.mat';
    
    % save result
    save(file_save,'ga'); 
    
end % End of GA


%% Reconstruction using searched parameters

% Load mask
file_mask = './data/db_tumor/mask/mask_tumor_125.mat';
load(file_mask);
mask = mask;

% Load 3D MRI
file_name = './data/db_tumor/data/tumor_3.mat';
load(file_name);
full = scale_img(full);


% Load searched parameters
GA_result_name = './searched_params/tumor/param_tumor_125.mat';
load(GA_result_name);
ga = ga;

% Undersample using mask
undersample = undersample_func(full, mask);

% options
opt.num_iters = 300; % number of iterations
opt.isDisplay = 1; % show figures
opt.isConsole = 1; % show outputs on console

% options data size
[x, y, z] = size(undersample);
opt.dataSizeX = x; 
opt.dataSizeY = y;
opt.dataSizeZ = z;

% Plot fullsample, undersample, mask
figure(1);
subplot(1,3,1);
imshow(full(:,:,1),[0 1]);
title('Fullsample');

temp = ifft2(undersample);
subplot(1,3,2);
imshow(temp(:,:,1),[0 1]);
title('Undersample');

subplot(1,3,3);
imshow(mask(:,:,1),[0 1]);
title('Mask');
drawnow();

% File save name
file_save = './result/recon_tumor_3.mat';

% Recontruct process
result = split_3DCSC_TV(full, mask, undersample, ga, opt);

% save result
save(file_save,'result'); 


