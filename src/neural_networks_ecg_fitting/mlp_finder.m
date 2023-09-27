clear;
close all;
clc;

%% Constants

N_REPETITION = 5;
MAX_HIDDEN_LAYER_NEURONS = 100;
MIN_HIDDEN_LAYER_NEURONS = 10;
HIDDEN_LAYER_NEURONS_STEP = 5;
ECG_TARGET = "mean"; % 'mean' or 'std'

addpath('./data_preprocessing');

%% Load Data and Initialize variables

load('../tmp/final_data');
if ECG_TARGET == "mean"
    x = final_features_ecg_mean_matrix';
    t = final_ecg_mean_targets_vector';
else
    x = final_features_ecg_std_matrix';
    t = final_ecg_std_targets_vector';
end

results = zeros((MAX_HIDDEN_LAYER_NEURONS - MIN_HIDDEN_LAYER_NEURONS) / HIDDEN_LAYER_NEURONS_STEP, N_REPETITION + 2);

%% MLP Training and Test

i = 1;
while true

    % Compute the new hidden layer size
    hidden_layer_size = MIN_HIDDEN_LAYER_NEURONS + (i - 1) * HIDDEN_LAYER_NEURONS_STEP;
    
    % Check if hidden_layer_size has reached the max value
    if hidden_layer_size > MAX_HIDDEN_LAYER_NEURONS
        break;
    end

    % Save the new hidden layer size
    results(i, 1) = hidden_layer_size;

    % Iterate all repetitions
    for j = 1 : N_REPETITION

        % Create and train a mlp network
        net = fitnet(hidden_layer_size, 'trainbr');
        net.trainParam.showWindow = 0;
        [net,tr] = train(net, x, t);

        % Test the network and save results
        y = net(x);
        % figure, plotregression(t, y);
        mse_value = mse(y', t');
        regression_stats = fitlm(t', y');
        r_value = sqrt(regression_stats.Rsquared.Ordinary);
        results(i, j + 1) = mse_value;
        results(i, j + 2) = r_value;

        fprintf("hidden neurons: %d, repetition: %d, mse: %d, r-value: %d\n", hidden_layer_size, j, mse_value, r_value);
    end
    
    i = i + 1;
end

if ECG_TARGET == "mean"
    writematrix(results, fullfile('../tmp', 'mlp_mean_ecg_fitting_results.csv'));
    save('../tmp/mlp_mean_ecg_fitting_results', 'results');
else
    writematrix(results, fullfile('../tmp', 'mlp_std_ecg_fitting_results.csv'));
    save('../tmp/mlp_std_ecg_fitting_results', 'results');
end
