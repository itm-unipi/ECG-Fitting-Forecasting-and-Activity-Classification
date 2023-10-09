clear;
close all;
clc;

%% Generate tmp folder if not exists

tmp_folder = '../tmp';
if ~exist(tmp_folder, 'dir')
    mkdir(tmp_folder);
end

%% Constants

RESOURCES_PATH = '../resources';
WINDOW_SIZE = 50000;
FRACTION_TEST_SET = 0.15;

N_CHANNELS = 12;
MAX_EPOCHS = 20;
MINI_BATCH_SIZE = 6;
HIDDEN_LAYER_SIZE = 200;
OUTPUT_LAYER_SIZE = 1;
INITIAL_LEARN_RATE = 0.01;

addpath('./recurrent_neural_network');
rng("default");

%% Generate the dataset

[dataset, targets] = get_dataset(RESOURCES_PATH, WINDOW_SIZE);

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

save('../tmp/rnn_final_dataset', ...
    'training_set', ...
    'training_targets', ...    
    'test_set', ...
    'test_targets');

%% Load

load('../tmp/rnn_final_dataset');

%% Define Network Architecture

layers = [ ...
    sequenceInputLayer(N_CHANNELS)
    
    lstmLayer(HIDDEN_LAYER_SIZE, 'OutputMode', 'sequence')
    
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(OUTPUT_LAYER_SIZE)
    
    regressionLayer
];

options = trainingOptions( ...
    'adam', ...
    ...
    MaxEpochs = MAX_EPOCHS, ...
    MiniBatchSize = MINI_BATCH_SIZE, ...
    Shuffle = 'never' , ...
    ...
    InitialLearnRate = INITIAL_LEARN_RATE, ....
    GradientThreshold = 1, ...
    ...
    ExecutionEnvironment = 'gpu', ...
    Plots = 'training-progress', ...
    Verbose = 1, ...
    VerboseFrequency = 1 ...
);

% Train the Network
net = trainNetwork(training_set, training_targets, layers, options);