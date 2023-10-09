clear;
close all;
clc;

%% Constants

RESOURCES_PATH = '../resources';
N_PREDICTIONS = 20; 

% Dataset generation parameters
WINDOW_SIZE = 50000;
FRACTION_TEST_SET = 0.15;

% Network layers parameters
N_CHANNELS = 12;
MAX_EPOCHS = 50;
MINI_BATCH_SIZE = 4;
LSMT_LAYER_SIZE = 100;
HIDDEN_LAYER_SIZE = 256;
DROPOUT_PROBABILITY = 0.3;

% Training options parameters
INITIAL_LEARN_RATE = 0.01;
LEARN_RATE_SCHEDULE = 'piecewise';
LEARN_RATE_DROP_PERIOD = 10;
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
    'test_targets');

%% Define RNN architecture and train it

load('../tmp/rnn_multi_step_final_dataset');

% Network structure
layers = [ ...
    sequenceInputLayer(N_CHANNELS)
    lstmLayer(LSMT_LAYER_SIZE)
    fullyConnectedLayer(HIDDEN_LAYER_SIZE)
    dropoutLayer(DROPOUT_PROBABILITY)
    fullyConnectedLayer(N_CHANNELS)
    regressionLayer
];

% Training option
options = trainingOptions( ...
    'adam', ...
    ...
    MaxEpochs = MAX_EPOCHS, ...
    MiniBatchSize = MINI_BATCH_SIZE, ...
    Shuffle = 'never' , ...
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

save('../tmp/rnn_multi_step_final_net', 'net_multi_step');

%% Multi-step Closed Loop Forecasting

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
        [net, y_predicted{i}(:, k)] = predictAndUpdateState(net, y_predicted{i}(:, k - 1));
    end
end

%% Test the RNN

targets = test_targets{1}(12, :);
x = 1:size(targets, 2);

figure;
plot(x(:, 1:10000), targets(:, 1:10000)');
hold on;
y = y_predicted{1}(12, :);
plot(x(:, 1:10000), y(:, 1:10000)');

% TODO: plot regression and RMSE



