clear;
close all;
clc;

%% Constants

MLP_HIDDEN_LAYER_SIZE = 35;
MLP_TRAINING_FUNCTION = "trainbr";

addpath('./neural_network_activity_classification');
addpath('./data_preprocessing');
rng("default");

%% Load Data and Initialize variables

load('../tmp/final_data');

x = final_features_activities_matrix';
t = full(ind2vec(final_activities_targets_vector'));

%% MLP Mean Training and Test

% Create and train a mlp network
net = patternnet(MLP_HIDDEN_LAYER_SIZE, MLP_TRAINING_FUNCTION);
[net, tr] = train(net, x, t);

% Test the network
train_x = x(:, tr.trainInd);
train_t = t(:, tr.trainInd);
train_y = net(train_x);
test_x = x(:, tr.testInd);
test_t = t(:, tr.testInd);
test_y = net(test_x);

% Show training results
figure(1), plotconfusion(train_t, train_y);
figure(2), plotroc(train_t, train_y);

% Show test results
figure(3), plotconfusion(test_t, test_y);
figure(4), plotroc(test_t, test_y);

% Save the figures
saveas(1, fullfile('../tmp', 'activities_classification_training_confusion_matrix.png'));
saveas(2, fullfile('../tmp', 'activities_classification_training_roc_analysis.png'));
saveas(3, fullfile('../tmp', 'activities_classification_test_confusion_matrix.png'));
saveas(4, fullfile('../tmp', 'activities_classification_test_roc_analysis.png'));

