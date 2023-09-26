clear;
close all;
clc;

%% Costants

MAX_NEURONS_NUMBER = 100;
NEURONS_NUMBER_STEP = 5;
SPREAD_STEP = 0.2;
N_REPETION = 5;
ERROR_GOAL = 0;
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

% Compute the spread range
distance = pdist(x');
min_spread = min(distance);
max_spread = max(distance);

fprintf("%d %d \n", min_spread, max_spread);
results = zeros(floor((max_spread - min_spread) / SPREAD_STEP), N_REPETION + 2);

%% RBF Training and Test 

i = 1;
spread = min_spread;
while true

    % Check if spread has reached the max value
    if(spread > max_spread)
        break;
    end

    % Save the new spread value
    results(i, 1) = spread;
    
    % Iterate all repetitions
    for j = 1 : N_REPETION

        % Create and train a rbf network with Bayesian regularization
        net = newrb(x, t, ERROR_GOAL, spread, MAX_NEURONS_NUMBER, NEURONS_NUMBER_STEP);
        net.trainFcn = 'trainbr';
        net.trainParam.showWindow = 0;
        net = train(net, x, t);

        % Test the network and save results
        y = net(x);
        % figure, plotregression(t, y);
        mse_value = mse(y', t');
        regression_stats = fitlm(t',y');
        r_value = sqrt(regression_stats.Rsquared.Ordinary);
        results(i, j + 1) = mse_value;
        results(i, j + 2) = r_value;

        fprintf("spread: %d, repetition: %d, mse: %d, r-value: %d\n", spread, j, mse_value, r_value);
    end

    i = i + 1;
    spread = spread + (max_spread - min_spread) * SPREAD_STEP;
end

if ECG_TARGET == "mean"
    writematrix(results, fullfile('../tmp', 'rbf_mean_ecg_fitting_results.csv'));
    save('../tmp/rbf_mean_ecg_fitting_results', results);
else
    writematrix(results, fullfile('../tmp', 'rbf_std_ecg_fitting_results.csv'));
    save('../tmp/rbf_std_ecg_fitting_results', results);
end

