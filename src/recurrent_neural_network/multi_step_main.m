clear;
close all;
clc;

%% Constants

RESOURCES_PATH = '../resources';
N_PREDICTIONS = 30; 

% Dataset generation parameters
WINDOW_SIZE = 500;
FRACTION_TEST_SET = 0.15;

% Network layers parameters
N_CHANNELS = 1;
MAX_EPOCHS = 30;
MINI_BATCH_SIZE = 32;
LSTM_LAYER_SIZE = 100;
HIDDEN_LAYER_SIZE = 200;
OUTPUT_LAYER_SIZE = 1;

% Training options parameters
INITIAL_LEARN_RATE = 0.01;
LEARN_RATE_SCHEDULE = 'piecewise';
LEARN_RATE_DROP_PERIOD = 10;
LEARN_RATE_DROP_FACTOR = 0.1;

addpath('./recurrent_neural_network');
rng("default");

figure_id = 1;

%% Generate the dataset

[dataset, targets] = get_dataset_multi_step(RESOURCES_PATH, WINDOW_SIZE);

% Normalize dataset with z-score
total_mean = mean([dataset{:}], 2);
total_std = std([dataset{:}], 0, 2);

for i = 1 : size(dataset, 1)
    dataset{i} = (dataset{i} - total_mean) ./ total_std;
end

% Generate training and test set
partition_data = cvpartition(size(dataset, 1), "Holdout", FRACTION_TEST_SET);

training_set = dataset(training(partition_data), :);
training_targets = targets(training(partition_data), :);

test_set = dataset(test(partition_data), :);
test_targets = targets(test(partition_data), :);

save('../tmp/rnn_multi_step_final_dataset', ...
    'training_set', ...
    'training_targets', ...    
    'test_set', ...
    'test_targets', ...
    'total_mean', ...
    'total_std');

%% Define RNN architecture and train it

load('../tmp/rnn_multi_step_final_dataset');

% Network structure
layers = [ ...
    sequenceInputLayer(N_CHANNELS)
    lstmLayer(LSTM_LAYER_SIZE)
    fullyConnectedLayer(HIDDEN_LAYER_SIZE)
    fullyConnectedLayer(N_CHANNELS)
    regressionLayer
];

% Training option
options = trainingOptions( ...
    'rmsprop', ...
    ...
    MaxEpochs = MAX_EPOCHS, ...
    MiniBatchSize = MINI_BATCH_SIZE, ...
    Shuffle = 'never', ...
    ...
    InitialLearnRate = INITIAL_LEARN_RATE, ...
    LearnRateSchedule = LEARN_RATE_SCHEDULE, ...
    LearnRateDropPeriod = LEARN_RATE_DROP_PERIOD, ...
    LearnRateDropFactor = LEARN_RATE_DROP_FACTOR, ...
    ...
    ExecutionEnvironment = 'gpu', ...
    Plots = 'training-progress', ...
    Verbose = 1, ...
    VerboseFrequency = 1 ...
);

net = trainNetwork(training_set, training_targets, layers, options);

save('../tmp/rnn_multi_step_final_net', 'net');

%% Multi-step Open Loop Forecasting

load('../tmp/rnn_multi_step_final_dataset');
load('../tmp/rnn_multi_step_final_net');

y_predicted = cell(size(test_set, 1), 1);

% Iterate all cells
for i = 1 : size(test_set, 1)

    offset = size(test_set{i}, 2) - N_PREDICTIONS; 

    % Initialize the RNN state using a portion of the dataset 
    net = resetState(net);
    y_predicted{i} = zeros(N_CHANNELS, size(test_set{i}, 2));
    [net, y_predicted{i}(:, 1 : offset)] = predictAndUpdateState(net, test_set{i}(:, 1 : offset));

    % Predict the subsequent values using new input and the RNN state 
    for k = offset + 1 : size(test_set{i}, 2)
        [net, y_predicted{i}(:, k)] = predictAndUpdateState(net, test_set{i}(:, k - 1));
    end

    fprintf("Forecasting the cell %d out of %d done\n", i, size(test_set, 1));
end

%% Multi-step Closed Loop Forecasting

load('../tmp/rnn_multi_step_final_dataset');
load('../tmp/rnn_multi_step_final_net');

y_predicted = cell(size(test_set, 1), 1);

% Iterate all cells
for i = 1 : size(test_set, 1)

    offset = size(test_set{i}, 2) - N_PREDICTIONS; 

    % Initialize the RNN state using a portion of the dataset 
    net = resetState(net);
    y_predicted{i} = zeros(N_CHANNELS, size(test_set{i}, 2));
    [net, y_predicted{i}(:, 1 : offset)] = predictAndUpdateState(net, test_set{i}(:, 1 : offset));

    % Predict the subsequent values using the previously predicted ones and the RNN state 
    for k = offset + 1 : size(test_set{i}, 2)
        y_normalized = (y_predicted{i}(:, k - 1) - total_mean) ./ total_std;
        [net, y_predicted{i}(:, k)] = predictAndUpdateState(net, y_normalized);
    end

    fprintf("Forecasting the cell %d out of %d done\n", i, size(test_set, 1));
end

%% Print cell forecasting preview

N_FIGURES = 5;
N_CELLS = 6;

% Iterate figures
for k = 1 : N_FIGURES

    figure(figure_id);

    % Iterate the cells of the figures
    for i = 1 : N_CELLS
  
        subplot(2, 3, i), grid on, hold on;

        % Plot the predicted values and the targets values
        plot(y_predicted{N_CELLS * (k - 1) + i}(:, end - N_PREDICTIONS + 1:end));
        plot(test_targets{N_CELLS * (k - 1) + i}(:, end - N_PREDICTIONS + 1:end));
    
        title("ECG of cell " + (N_CELLS * (k - 1) + i)) ;
    end

    saveas(figure_id, "../tmp/rnn_multi_step_preview_" + k + ".png");
    figure_id = figure_id + 1;
end

%% Print a specific cell

cell = 10;

figure(figure_id), grid on, hold on;
plot(y_predicted{cell}(:, end - N_PREDICTIONS + 1:end), 'Marker', 'o', 'MarkerSize', 5);
plot(test_targets{cell}(:, end - N_PREDICTIONS + 1:end), 'Marker', 'o', 'MarkerSize', 5);
title("Predict ECG of cell " + cell) ;

saveas(figure_id, "../tmp/rnn_multi_step_cell_" + cell  + ".png");
figure_id = figure_id + 1;

%% Compute the mean RMSE of all predicted cells

rmse = zeros(1, size(y_predicted, 1));

for i = 1 : size(y_predicted, 1)
    rmse(1, i) = sqrt(mean((y_predicted{i}(1, end - N_PREDICTIONS + 1:end) - test_targets{i}(1, end - N_PREDICTIONS + 1:end)).^2, "all"));
end

fprintf("Mean RMSE for all cells: %d\n", (mean(rmse)));
