clear;
close all;
clc;

%% Costants

MAX_NEURONS_NUMBER = 100;
NEURONS_NUMBER_STEP = 5;
SPREAD_STEP = 0.2;
N_REPETION = 5;
ECG_TARGET = "mean"; % 'mean' or 'std'

addpath('./data_preprocessing');

%% Load Data and Initialize variables

load('../tmp/final_data');
x = final_features_ecg_mean_matrix';
if ECG_TARGET == "mean"
    t = final_ecg_mean_targets_vector';
else
    t = final_ecg_std_targets_vector';
end

% Compute the spread range
distance = pdist(x);
min_spread = min(distance);
max_spread = max(distance);

results = zeros(floor((max_spread - min_spread) / SPREAD_STEP), N_REPETION + 1);

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
        net = newrbe(x, t, spread, MAX_NEURONS_NUMBER, NEURONS_NUMBER_STEP);
        net.trainFcn = 'trainbr';
        net = train(net, x, t);

        % Test the network and save results
        y = net(x);
        % figure, plotregression(t, y);
        regression_stats = fitlm(t',y');
        r_value = sqrt(regression_stats.Rsquared.Ordinary);
        results(i, j + 1) = r_value;

        fprintf("spread: %d, repetition: %d, r-value: %d\n", spread, j, r_value);
    end

    i = i + 1;
    spread = spread + (max_spread - min_spread) * SPREAD_STEP;
end

if ECG_TARGET == "mean"
    save('../tmp/rbf_mean_ecg_fitting_results', results);
else
    save('../tmp/rbf_std_ecg_fitting_results', results);
end

