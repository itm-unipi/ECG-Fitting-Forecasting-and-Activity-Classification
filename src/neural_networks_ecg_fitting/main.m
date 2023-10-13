clear;
close all;
clc;

%% Constants

% MLP constants
MLP_MEAN_HIDDEN_LAYER_SIZE = 20;
MLP_STD_HIDDEN_LAYER_SIZE = 20;

% RBF Costants
MAX_NEURONS_NUMBER = 100;
NEURONS_NUMBER_STEP = 5;
SPREAD_MEAN = 0.8;
SPREAD_STD = 0.8;
ERROR_GOAL = 0;

addpath('./data_preprocessing');
rng("default");

%% Load Data and Initialize variables

load('../tmp/final_data');

x_mean = final_features_ecg_mean_matrix';
t_mean = final_ecg_mean_targets_vector';
x_std = final_features_ecg_std_matrix';
t_std = final_ecg_std_targets_vector';

%% MLP Mean Training and Test

% Create and train a mlp network
net = fitnet(MLP_MEAN_HIDDEN_LAYER_SIZE, 'trainbr');
[net, ~] = train(net, x_mean, t_mean);

% Test the network and save results
y = net(x_mean);
figure_id = 1;
figure(figure_id), plotregression(y, t_mean);
mse_value = mse(y', t_mean');
regression_stats = fitlm(y', t_mean');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("MLP MEAN RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'mlp_mean_result.png'));

%% MLP Std Training and Test

clear net;
clear y;

% Create and train a mlp network
net = fitnet(MLP_STD_HIDDEN_LAYER_SIZE, 'trainbr');
[net, ~] = train(net, x_std, t_std);

% Test the network and save results
y = net(x_std);
figure_id = figure_id + 1;
figure(figure_id), plotregression(y, t_std);
mse_value = mse(y', t_std');
regression_stats = fitlm(y', t_std');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("MLP STD RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'mlp_std_result.png'));

%% RBF Mean Training and test

clear net;
clear y;

% Create and train a rbf network with Bayesian regularization
net = newrb(x_mean, t_mean, ERROR_GOAL, SPREAD_MEAN, MAX_NEURONS_NUMBER, NEURONS_NUMBER_STEP);
net.trainFcn = 'trainbr';
net = train(net, x_mean, t_mean);

% Test the network and save results
y = net(x_mean);
figure_id = figure_id + 1;
figure(figure_id), plotregression(y, t_mean);
mse_value = mse(y', t_mean');
regression_stats = fitlm(y', t_mean');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("RBF MEAN RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'rbf_mean_result.png'));

%% RBF Std Training and test

clear net;
clear y;

% Create and train a rbf network with Bayesian regularization
net = newrb(x_std, t_std, ERROR_GOAL, SPREAD_STD, MAX_NEURONS_NUMBER, NEURONS_NUMBER_STEP);
net.trainFcn = 'trainbr';
net = train(net, x_std, t_std);

% Test the network and save results
y = net(x_std);
figure_id = figure_id + 1;
figure(figure_id), plotregression(y, t_std);
mse_value = mse(y', t_std');
regression_stats = fitlm(y', t_std');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("RBF STD RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'rbf_std_result.png'));
