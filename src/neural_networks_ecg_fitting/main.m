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
FRACTION_RBF_TEST_SET = 0.15;
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
[net, tr] = train(net, x_mean, t_mean);

% Test the network and save results
test_x_mean = x_mean(:, tr.testInd);
test_t_mean = t_mean(:, tr.testInd);
test_y_mean = net(test_x_mean);
figure_id = 1;
figure(figure_id), plotregression(test_t_mean, test_y_mean);
mse_value = mse(test_t_mean', test_y_mean');
regression_stats = fitlm(test_t_mean', test_y_mean');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("MLP MEAN RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'mlp_mean_result.png'));

%% MLP Std Training and Test

clear net;
clear y;

% Create and train a mlp network
net = fitnet(MLP_STD_HIDDEN_LAYER_SIZE, 'trainbr');
[net, tr] = train(net, x_std, t_std);

% Test the network and save results
test_x_std = x_std(:, tr.testInd);
test_t_std = t_std(:, tr.testInd);
test_y_std = net(test_x_std);
figure_id = figure_id + 1;
figure(figure_id), plotregression(test_t_std, test_y_std);
mse_value = mse(test_t_std', test_y_std');
regression_stats = fitlm(test_t_std', test_y_std');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("MLP STD RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'mlp_std_result.png'));

%% Dataset partitioning for RBF

clear training_x_mean;
clear training_t_mean;
clear test_x_mean;
clear test_t_mean;
clear training_x_std;
clear training_t_std;
clear test_x_std;
clear test_t_std;

partition_data = cvpartition(size(x_mean, 2), "Holdout", FRACTION_RBF_TEST_SET);

% Mean dataset
training_x_mean = x_mean(:, training(partition_data));
training_t_mean = t_mean(:, training(partition_data));
test_x_mean = x_mean(:, test(partition_data));
test_t_mean = t_mean(:, test(partition_data));

partition_data = cvpartition(size(x_std, 2), "Holdout", FRACTION_RBF_TEST_SET);

% Std dataset
training_x_std = x_std(:, training(partition_data));
training_t_std = t_std(:, training(partition_data));
test_x_std = x_std(:, test(partition_data));
test_t_std = t_std(:, test(partition_data));

%% RBF Mean Training and test

clear net;

% Create and train a rbf network with Bayesian regularization
net = newrb(training_x_mean, training_t_mean, ERROR_GOAL, SPREAD_MEAN, MAX_NEURONS_NUMBER, NEURONS_NUMBER_STEP);
net.trainFcn = 'trainbr';
net = train(net, training_x_mean, training_t_mean);

% Test the network and save results
test_y_mean = net(test_x_mean);
figure_id = figure_id + 1;
figure(figure_id), plotregression(test_t_mean, test_y_mean);
mse_value = mse(test_t_mean', test_y_mean');
regression_stats = fitlm(test_t_mean', test_y_mean');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("RBF MEAN RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'rbf_mean_result.png'));

%% RBF Std Training and test

clear net;

% Create and train a rbf network with Bayesian regularization
net = newrb(training_x_std, training_t_std, ERROR_GOAL, SPREAD_STD, MAX_NEURONS_NUMBER, NEURONS_NUMBER_STEP);
net.trainFcn = 'trainbr';
net = train(net, training_x_std, training_t_std);

% Test the network and save results
test_y_std = net(test_x_std);
figure_id = figure_id + 1;
figure(figure_id), plotregression(test_t_std, test_y_std);
mse_value = mse(test_t_std', test_y_std');
regression_stats = fitlm(test_t_std', test_y_std');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("RBF STD RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'rbf_std_result.png'));
