clear;
close all;
clc;

%% Constants

MLP_HIDDEN_LAYER_SIZE = 20;
MLP_TRAINING_FUNCTION = "trainbr";

addpath('./data_preprocessing');

%% Load Data and Initialize variables

load('../tmp/final_data');

x = final_features_activities_matrix';
t = full(ind2vec(final_activities_targets_vector'));

%% MLP Mean Training and Test

% Create and train a mlp network
net = patternnet(MLP_HIDDEN_LAYER_SIZE, MLP_TRAINING_FUNCTION);
[net, tr] = train(net, x, t);

% Test the network
test_x = x(:, tr.testInd);
test_t = t(:, tr.testInd);
test_y = net(test_x);
[c, ~] = confusion(test_t, test_y);
correct_classification_percentage = 100 * (1 - c);
incorrect_classification_percentage = 100 * c;

% Show results
figure(1), plotconfusion(test_t, test_y);
figure(2), plotroc(test_t, test_y);
fprintf('Percentage Correct Classification: %f%%\n', correct_classification_percentage);
fprintf('Percentage Incorrect Classification: %f%%\n', incorrect_classification_percentage);

% Save the figures
saveas(1, fullfile('../tmp', 'activities_classification_confusion_matrix.png'));
saveas(2, fullfile('../tmp', 'activities_classification_roc_analysis.png'));

