clear;
close all;
clc;

%% Constants

RESOURCES_PATH = '../resources';
N_PREDICTIONS = 50; 

% Dataset generation parameters
WINDOW_SIZE = 200;
FRACTION_TEST_SET = 0.15;

% Network layers parameters
N_CHANNELS = 1;
MAX_EPOCHS = 21;
MINI_BATCH_SIZE = 9;
LSTM_LAYER_SIZE = 64;
HIDDEN_LAYER_SIZE = 256;
OUTPUT_LAYER_SIZE = 1;
% DROPOUT_PROBABILITY = 0.4;

% Training options parameters
INITIAL_LEARN_RATE = 0.1;
LEARN_RATE_SCHEDULE = 'piecewise';
LEARN_RATE_DROP_PERIOD = 7;
LEARN_RATE_DROP_FACTOR = 0.1;

addpath('./recurrent_neural_network');
rng("default");

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
    'adam', ...
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
    SequencePaddingDirection = 'left', ...
    ExecutionEnvironment = 'gpu', ...
    Plots = 'training-progress', ...
    Verbose = 1, ...
    VerboseFrequency = 1 ...
);

net = trainNetwork(training_set, table2array(training_targets), layers, options);

save('../tmp/rnn_multi_step_final_net', 'net');

%% Multi-step Closed Loop Forecasting

load('../tmp/rnn_multi_step_final_net')

y_predicted = cell(size(test_set, 1), 1);

% Iterate all cells
for i = 1 : size(test_set, 1)

    offset = size(test_set{i}, 2) - N_PREDICTIONS; 

    % Initialize the RNN state using a portion of the dataset 
    net = resetState(net);
    y_predicted{i} = zeros(N_CHANNELS, size(test_set{i}, 2));
    [net, y_predicted{i}(:, 1 : offset)] = predictAndUpdateState(net, test_set{i}(:, 1:offset));

    % Predict the subsequent values using the previously predicted ones and the RNN state 
    for k = offset + 1 : size(test_set{i}, 2)
        y_normalized = (y_predicted{i}(:, k - 1) - total_mean) ./ total_std;
        [net, y_predicted{i}(:, k)] = predictAndUpdateState(net, y_normalized);
    end

    fprintf("%d/%d\n", i, size(test_set, 1));
end

%% Test the RNN

% Compare predictions of 4 random cells with original targets
cell_index = randperm(size(y_predicted, 1), 4);

figure(1);
for i = 1: size(cell_index, 2)

    targets = test_targets{i}(end, :);
    x = 1:size(targets, 2);

    subplot(2, 2, i);
    plot(x(:, (end - N_PREDICTIONS) : end), targets(:, (end - N_PREDICTIONS) : end)');
    grid on;
    hold on;
    y = y_predicted{1}(end, :);
    plot(x(:, (end - N_PREDICTIONS) : end), y(:, (end - N_PREDICTIONS) : end)');
    title("ECG for cell " + cell_index(1, i));
end

saveas(1, "../tmp/rnn_multi_step_predictions.png");

%% Plot regression and RMSE

all_test_targets = cat(2, test_targets{:});
all_y_predicted = cat(2, y_predicted{:});

% only for ECG
figure(2); 
plotregression(all_test_targets(1, end - N_PREDICTIONS: end), all_y_predicted(1, end - N_PREDICTIONS: end));
saveas(2, '../tmp/rnn_multi_step_test_regression.png');

rmse = zeros(N_CHANNELS, size(y_predicted, 1));

% Compute the RMSE for each channel and cell
for i = 1 : size(y_predicted, 1)
    for k = 1 : N_CHANNELS
        rmse(k, i) = sqrt(mean((y_predicted{i}(k, :) - test_targets{i}(k, :)).^2, "all"));
    end
end

% Generate RMSE plot for all channels
figure(3);
for i = 1 : N_CHANNELS

    subplot(3, 4, i);

    stem(rmse(i, :));
    grid on;
    ylabel("RMSE");
    xlabel("# Test");
    title("RMSE of each test (ch " + i + ")");

    % Calculate the mean RMSE
    fprintf("Mean RMSE (channel %d): %d\n", i, mean(rmse(i, :)));
end

saveas(3, "../tmp/rnn_multi_step_rmse.png");

% Generate RMSE plot for ecg
figure(4);
stem(rmse(end, :));
grid on;
ylabel("RMSE");
xlabel("# Test");
title("RMSE of each test (ECG)");
saveas(4, "../tmp/rnn_multi_step_rmse_ecg.png");