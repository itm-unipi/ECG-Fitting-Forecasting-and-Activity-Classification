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
WINDOW_SIZE = 50000;
FRACTION_TEST_SET = 0.15;

% Network layers parameters
N_CHANNELS = 12;
MAX_EPOCHS = 50;
MINI_BATCH_SIZE = 4;
LSMT_LAYER_SIZE = 100;
HIDDEN_LAYER_SIZE = 256;
OUTPUT_LAYER_SIZE = 1;
DROPOUT_PROBABILITY = 0.3;

% Training options parameters
INITIAL_LEARN_RATE = 0.01;
LEARN_RATE_SCHEDULE = 'piecewise';
LEARN_RATE_DROP_PERIOD = 10;
LEARN_RATE_DROP_FACTOR = 0.1;

addpath('./recurrent_neural_network');
rng("default");

%% Generate the dataset

[dataset, targets] = get_dataset_single_step(RESOURCES_PATH, WINDOW_SIZE);

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

%% Define RNN architecture and train it

load('../tmp/rnn_final_dataset');

% Network structure
layers = [ ...
    sequenceInputLayer(N_CHANNELS)
    lstmLayer(LSMT_LAYER_SIZE)
    fullyConnectedLayer(HIDDEN_LAYER_SIZE)
    dropoutLayer(DROPOUT_PROBABILITY)
    fullyConnectedLayer(OUTPUT_LAYER_SIZE)
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

save('../tmp/rnn_single_step_final_net', 'net');

%% Test the RNN

y_training = predict(net, training_set, ExecutionEnvironment='gpu', MiniBatchSize=6);
y_test = predict(net, test_set, ExecutionEnvironment='gpu', MiniBatchSize=6);

targets = test_targets{1};
x = 1:size(targets, 2);
y = y_test{1};

figure;
plot(x(:, 1:10000), targets(:, 1:10000)');
hold on;
plot(x(:, 1:10000), y(:, 1:10000)');

% Plot regression and save result
figure(1); 
plotregression(training_targets, y_training);
saveas(1, '../tmp/rnn_single_step_training_regression.png');
figure(2); 
plotregression(test_targets, y_test);
saveas(2, '../tmp/rnn_single_step_test_regression.png');

% Compute the RMSE
rmse = zeros(1, size(y_test, 1));

for i = 1 : size(y_test, 1)
    rmse(i) = sqrt(mean((y_test{i} - test_targets{i}).^2, "all"));
end

figure(3);
stem(rmse);
grid on;
ylabel("RMSE");
xlabel("# Test");
title("RMSE of each test");
saveas(3, '../tmp/rnn_single_step_rmse.png');

% Calculate the mean RMSE over all test observations
fprintf("Mean RMSE: %d\n", mean(rmse));