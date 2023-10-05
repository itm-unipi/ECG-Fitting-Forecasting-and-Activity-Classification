clear;
close all;
clc;

%% Constants

RESOURCES_PATH = '../resources';
WINDOW_SIZE = 5000;
FRACTION_TEST_SET = 0.15;
N_CHANNELS = 11;
N_FILTERS = 125;
FILTER_SIZE = 10;
STRIDE = 2;
HIDDEN_LAYER_SIZE = 100;
OUTPUT_LAYER_SIZE = 1;
MAX_EPOCHS = 50;
MINI_BATCH_SIZE = 64;

addpath('./convolutional_neural_network');
rng("default");

%% Generate the dataset and remove outliers

[dataset, targets] = get_dataset(RESOURCES_PATH, WINDOW_SIZE);

% Remove outliers 
[targets, outliers] = rmoutliers(targets);
dataset = dataset(~outliers);

% Generate training and test set
partition_data = cvpartition(size(dataset, 1), "Holdout", FRACTION_TEST_SET);

training_set = dataset(training(partition_data), :);
training_targets = targets(training(partition_data), :);

test_set = dataset(test(partition_data), :);
test_targets = targets(test(partition_data), :);

save('../tmp/cnn_final_dataset', ...
    'training_set', ...
    'training_targets', ...    
    'test_set', ...
    'test_targets');

%%

load('../tmp/cnn_final_dataset_5000');

%% Define CNN architecture and train it

layers = [
    sequenceInputLayer(N_CHANNELS)

    convolution1dLayer(FILTER_SIZE, N_FILTERS, 'Stride', STRIDE, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer  
    maxPooling1dLayer(4, 'Stride', 4, 'Padding', 'same')

    convolution1dLayer(FILTER_SIZE, N_FILTERS * 2, 'Stride', STRIDE, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer  
    maxPooling1dLayer(4, 'Stride', 4, 'Padding', 'same')

    convolution1dLayer(FILTER_SIZE, N_FILTERS * 3, 'Stride', STRIDE, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer  
    maxPooling1dLayer(4, 'Stride', 4, 'Padding', 'same')

    convolution1dLayer(FILTER_SIZE, N_FILTERS * 4, 'Stride', STRIDE, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer  
    maxPooling1dLayer(4, 'Stride', 4, 'Padding', 'same')

    globalAveragePooling1dLayer
    fullyConnectedLayer(HIDDEN_LAYER_SIZE)
    fullyConnectedLayer(OUTPUT_LAYER_SIZE)
    
    regressionLayer
];

options = trainingOptions( ...
    'adam', ...
    ...
    MaxEpochs = MAX_EPOCHS, ...
    MiniBatchSize = MINI_BATCH_SIZE, ...
    Shuffle = 'every-epoch' , ...
    ...
    InitialLearnRate = 0.01, ...
    LearnRateSchedule = 'piecewise', ...
    LearnRateDropPeriod = 10, ...
    LearnRateDropFactor = 0.1, ...
    L2Regularization = 0.01, ...
    ...
    ExecutionEnvironment = 'gpu', ...
    Plots = 'training-progress', ...
    Verbose = 1, ...
    VerboseFrequency = 1 ...
);

net = trainNetwork(training_set, training_targets, layers, options);

%% Test the CNN

y_training = predict(net, training_set, ExecutionEnvironment='gpu');
y_test = predict(net, test_set, ExecutionEnvironment='gpu');

% Plot regression 
figure; 
plotregression(training_targets, y_training);
figure; 
plotregression(test_targets, y_test);

% layer_graph = layerGraph(layers);

