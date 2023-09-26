clear;
close all;
clc;

%% Constants

MLP_MEAN_HIDDEN_LAYER_SIZE = 20;
MLP_STD_HIDDEN_LAYER_SIZE = 20;

addpath('./data_preprocessing');

%% Load Data and Initialize variables

load('../tmp/final_data');

x_mean = final_features_ecg_mean_matrix';
t_mean = final_ecg_mean_targets_vector';
x_std = final_features_ecg_std_matrix';
t_std = final_ecg_std_targets_vector';

%% MLP Mean Training and Test

% Create and train a mlp network
net_mean = fitnet(MLP_MEAN_HIDDEN_LAYER_SIZE, 'trainbr');
net_mean.trainParam.showWindow = 0;
[net_mean, tr] = train(net_mean, x_mean, t_mean);

% Test the network and save results
y_mean = net_mean(x_mean);
figure_id = 1;
figure(figure_id), plotregression(t_mean, y_mean);
mse_value = mse(y_mean', t_mean');
regression_stats = fitlm(t_mean', y_mean');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("MLP MEAN RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'mlp_mean_result.png'));

%% MLP Std Training and Test

% Create and train a mlp network
net_std = fitnet(MLP_MEAN_HIDDEN_LAYER_SIZE, 'trainbr');
net_std.trainParam.showWindow = 0;
[net_std, tr] = train(net_std, x_std, t_std);

% Test the network and save results
y_std = net_std(x_std);
figure_id = figure_id + 1;
figure(figure_id), plotregression(t_std, y_std);
mse_value = mse(y_std', t_std');
regression_stats = fitlm(t_std', y_std');
r_value = sqrt(regression_stats.Rsquared.Ordinary);

fprintf("MLP MEAN RESULTS: [mse: %d, r-value: %d]\n", mse_value, r_value);

saveas(figure_id, fullfile('../tmp', 'mlp_std_result.png'));
