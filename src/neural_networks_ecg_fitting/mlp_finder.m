clear;
close all;
clc;

%% Constants

N_REPETITION = 5;
MAX_HIDDEN_LAYER_NEURONS = 100;
MIN_HIDDEN_LAYER_NEURONS = 10;
HIDDEN_LAYER_NEURONS_STEP = 5;
ECG_TARGETS = ["mean", "std"];
MAX_EPOCHS = [200, 350, 500];

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
    
    results = zeros(floor((MAX_HIDDEN_LAYER_NEURONS - MIN_HIDDEN_LAYER_NEURONS) / HIDDEN_LAYER_NEURONS_STEP) * size(MAX_EPOCHS, 2), N_REPETITION * 2 + 2);

    % MLP Training and Test
    i = 1;
    while true
    
        % Compute the new hidden layer size
        hidden_layer_size = MIN_HIDDEN_LAYER_NEURONS + (i - 1) * HIDDEN_LAYER_NEURONS_STEP;
        
        % Check if hidden_layer_size has reached the max value
        if hidden_layer_size > MAX_HIDDEN_LAYER_NEURONS
            break;
        end
        
        % Iterate different epochs
        for m = 1 : size(MAX_EPOCHS, 2)

            row_index = (i - 1) * size(MAX_EPOCHS, 2) + m;

            % Save the hidden layer size and max epoch
            results(row_index, 1) = hidden_layer_size;
            results(row_index, 2) = MAX_EPOCHS(m);
        
            % Iterate all repetitions
            for j = 1 : N_REPETITION
        
                % Create and train a mlp network
                net = fitnet(hidden_layer_size, 'trainbr');
                net.trainParam.showWindow = 0;
                net.trainParam.epochs = MAX_EPOCHS(m);
                net.divideParam.trainRatio = 0.85;
                net.divideParam.testRatio = 0.15;
                net.divideParam.valRatio = 0;
                [net, tr] = train(net, x, t);
        
                % Test the network and save results
                train_x = x(:, tr.trainInd);
                train_t = t(:, tr.trainInd);
                train_y = net(train_x);
                test_x = x(:, tr.testInd);
                test_t = t(:, tr.testInd);
                test_y = net(test_x);
                train_regression_stats = fitlm(train_t', train_y');
                test_regression_stats = fitlm(test_t', test_y');
                train_r_value = sqrt(train_regression_stats.Rsquared.Ordinary);
                test_r_value = sqrt(test_regression_stats.Rsquared.Ordinary);
                results(row_index, j * 2 + 1) = train_r_value;
                results(row_index, j * 2 + 2) = test_r_value;
        
                fprintf("hidden neurons: %d, epochs: %d, repetition: %d, training r-value: %d, test r-value: %d\n", hidden_layer_size, MAX_EPOCHS(m), j, train_r_value, test_r_value);
            end
        end
        
        i = i + 1;
    end
    
    if ECG_TARGETS(k) == "mean"
        writematrix(results, fullfile('../tmp', 'mlp_mean_ecg_fitting_results.csv'));
        save('../tmp/mlp_mean_ecg_fitting_results', 'results');
    else
        writematrix(results, fullfile('../tmp', 'mlp_std_ecg_fitting_results.csv'));
        save('../tmp/mlp_std_ecg_fitting_results', 'results');
    end
end