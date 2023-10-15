clear;
close all;
clc;

%% Costants

MAX_NEURONS_NUMBER = 100;
NEURONS_NUMBER_STEP = 5;
SPREAD_STEP = 0.2;
N_REPETITION = 1;
ERROR_GOAL = 0;
ECG_TARGETS = ["mean", "std"];
MAX_EPOCHS = [200, 350, 500];
FRACTION_TEST_SET = 0.15;

addpath('./data_preprocessing');
rng("default");

% Iterate ecg targets
for k = 1 : size(ECG_TARGETS, 2)

    % Load Data and Initialize variables
    load('../tmp/final_data');
    if ECG_TARGETS(k) == "mean"
        x = final_features_ecg_mean_matrix';
        t = final_ecg_mean_targets_vector';
    else
        x = final_features_ecg_std_matrix';
        t = final_ecg_std_targets_vector';
    end
    
    % Partition dataset
    partition_data = cvpartition(size(x, 2), "Holdout", FRACTION_TEST_SET);
    train_x = x(:, training(partition_data));
    train_t = t(:, training(partition_data));
    test_x = x(:, test(partition_data));
    test_t = t(:, test(partition_data));
    
    % Compute the spread range
    distance = pdist(train_x');
    min_spread = min(distance);
    max_spread = max(distance);
    
    fprintf("%d %d \n", min_spread, max_spread);
    results = zeros(floor(1 / SPREAD_STEP) * size(MAX_EPOCHS, 2), N_REPETITION * 2 + 2);
    
    % RBF Training and Test 
    
    i = 1;
    spread = min_spread;
    while true
    
        % Check if spread has reached the max value
        if(spread > max_spread)
            break;
        end
        
        % Iterate different epochs
        for m = 1 : size(MAX_EPOCHS, 2)

            row_index = (i - 1) * size(MAX_EPOCHS, 2) + m;

            % Save the new spread value and max epoch
            results(row_index, 1) = spread;
            results(row_index, 2) = MAX_EPOCHS(m);

            % Iterate all repetitions
            for j = 1 : N_REPETITION
        
                % Create and train a rbf network with Bayesian regularization
                net = newrb(train_x, train_t, ERROR_GOAL, spread, MAX_NEURONS_NUMBER, NEURONS_NUMBER_STEP);
                net.trainFcn = 'trainbr';
                net.trainParam.showWindow = 0;
                net.trainParam.epochs = MAX_EPOCHS(m);
                net = train(net, train_x, train_t);
        
                % Test the network and save results
                train_y = net(train_x);
                test_y = net(test_x);
                train_regression_stats = fitlm(train_t', train_y');
                test_regression_stats = fitlm(test_t', test_y');
                train_r_value = sqrt(train_regression_stats.Rsquared.Ordinary);
                test_r_value = sqrt(test_regression_stats.Rsquared.Ordinary);
                results(row_index, j * 2 + 1) = train_r_value;
                results(row_index, j * 2 + 2) = test_r_value;
        
                fprintf("spread: %d, epoch: %d, repetition: %d, training r-value: %d, test r-value: %d\n", spread, MAX_EPOCHS(m), j, train_r_value, test_r_value);
            end
        end

        i = i + 1;
        spread = spread + (max_spread - min_spread) * SPREAD_STEP;
    end
    
    if ECG_TARGETS(k) == "mean"
        writematrix(results, fullfile('../tmp', 'rbf_mean_ecg_fitting_results.csv'));
        save('../tmp/rbf_mean_ecg_fitting_results', 'results');
    else
        writematrix(results, fullfile('../tmp', 'rbf_std_ecg_fitting_results.csv'));
        save('../tmp/rbf_std_ecg_fitting_results', 'results');
    end
end

