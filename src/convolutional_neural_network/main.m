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

% Dataset generation parameters
WINDOW_SIZE = 5000;
FRACTION_TEST_SET = 0.15;
FRACTION_VALIDATION_SET = 0.117;

% Network layers parameters
N_CHANNELS = 11;
N_FILTERS = [96 256 348 348 256];
FILTER_SIZE = [11 5 3 3 3];
CONVOLUTIONAL_STRIDE = [4 4 4 4 4];
POOLING_COMPRESSION = [2 2 2];
POOLING_STRIDE = [2 2 2];
HIDDEN_LAYER_SIZE = 256;
OUTPUT_LAYER_SIZE = 1;

% Training options parameters
MAX_EPOCHS = 80;
MINI_BATCH_SIZE = 64;
INITIAL_LEARN_RATE = 0.01;
LEARN_RATE_SCHEDULE = 'piecewise';
LEARN_RATE_DROP_PERIOD = 20;
LEARN_RATE_DROP_FACTOR = 0.1;

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

%% Define CNN architecture and train it

load('../tmp/cnn_final_dataset');

% Network structure
layers = [
    sequenceInputLayer(N_CHANNELS)

    convolution1dLayer(FILTER_SIZE(1), N_FILTERS(1), 'Stride', CONVOLUTIONAL_STRIDE(1), 'Padding', 'same')
    batchNormalizationLayer
    maxPooling1dLayer(POOLING_COMPRESSION(1), 'Stride', POOLING_STRIDE(1), 'Padding', 'same')
    reluLayer

    convolution1dLayer(FILTER_SIZE(2), N_FILTERS(2), 'Stride', CONVOLUTIONAL_STRIDE(2), 'Padding', 'same')
    batchNormalizationLayer
    maxPooling1dLayer(POOLING_COMPRESSION(2), 'Stride', POOLING_STRIDE(2), 'Padding', 'same')
    reluLayer

    convolution1dLayer(FILTER_SIZE(3), N_FILTERS(3), 'Stride', CONVOLUTIONAL_STRIDE(3), 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(FILTER_SIZE(4), N_FILTERS(4), 'Stride', CONVOLUTIONAL_STRIDE(4), 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution1dLayer(FILTER_SIZE(5), N_FILTERS(5), 'Stride', CONVOLUTIONAL_STRIDE(5), 'Padding', 'same')
    batchNormalizationLayer
    maxPooling1dLayer(POOLING_COMPRESSION(3), 'Stride', POOLING_STRIDE(3), 'Padding', 'same')
    reluLayer

    globalAveragePooling1dLayer
    fullyConnectedLayer(HIDDEN_LAYER_SIZE)
    fullyConnectedLayer(OUTPUT_LAYER_SIZE)

    regressionLayer
];

% Training option
options = trainingOptions( ...
    'adam', ...
    ...
    MaxEpochs = MAX_EPOCHS, ...
    MiniBatchSize = MINI_BATCH_SIZE, ...
    Shuffle = 'every-epoch' , ...
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

%% Test the CNN

y_training = predict(net, training_set, ExecutionEnvironment='gpu');
y_test = predict(net, test_set, ExecutionEnvironment='gpu');

% Plot regression and save result
figure(1); 
plotregression(training_targets, y_training);
saveas(1, '../tmp/cnn_training_regression.png');
figure(2); 
plotregression(test_targets, y_test);
saveas(2, '../tmp/cnn_test_regression.png');
