clear;
close all;
clc;

%% Costants

MAX_NEURONS_NUMBER = 100;
NEURONS_NUMBER_STEP = 5;
SPREAD_STEP = 0.2;
N_REPETITION = 5;
ERROR_GOAL = 0;
ECG_TARGET = "std"; % 'mean' or 'std'
FRACTION_TEST_SET = 0.15;

addpath('./data_preprocessing');
rng("default");

%% Load Data and Initialize variables

load('../tmp/final_data');
if ECG_TARGET == "mean"
    x = final_features_ecg_mean_matrix';
    t = final_ecg_mean_targets_vector';
else
    x = final_features_ecg_std_matrix';
    t = final_ecg_std_targets_vector';
end

% Partition dataset
partition_data = cvpartition(size(x, 2), "Holdout", FRACTION_TEST_SET);
training_x = x(:, training(partition_data));
training_t = t(:, training(partition_data));
test_x = x(:, test(partition_data));
test_t = t(:, test(partition_data));

% Compute the spread range
distance = pdist(training_x');
min_spread = min(distance);
max_spread = max(distance);

fprintf("%d %d \n", min_spread, max_spread);
results = zeros(floor(1 / SPREAD_STEP), N_REPETITION * 2 + 1);

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
    for j = 1 : N_REPETITION

        % Create and train a rbf network with Bayesian regularization
        net = newrb(training_x, training_t, ERROR_GOAL, spread, MAX_NEURONS_NUMBER, NEURONS_NUMBER_STEP);
        net.trainFcn = 'trainbr';
        net.trainParam.showWindow = 0;
        net = train(net, training_x, training_t);

        % Test the network and save results
        test_y = net(test_x);
        % figure, plotregression(t, y);
        mse_value = mse(test_t', test_y');
        regression_stats = fitlm(test_t', test_y');
        r_value = sqrt(regression_stats.Rsquared.Ordinary);
        results(i, j * 2) = mse_value;
        results(i, j * 2 + 1) = r_value;

        fprintf("spread: %d, repetition: %d, mse: %d, r-value: %d\n", spread, j, mse_value, r_value);
    end

    i = i + 1;
    spread = spread + (max_spread - min_spread) * SPREAD_STEP;
end

if ECG_TARGET == "mean"
    writematrix(results, fullfile('../tmp', 'rbf_mean_ecg_fitting_results.csv'));
    save('../tmp/rbf_mean_ecg_fitting_results', 'results');
else
    writematrix(results, fullfile('../tmp', 'rbf_std_ecg_fitting_results.csv'));
    save('../tmp/rbf_std_ecg_fitting_results', 'results');
end

