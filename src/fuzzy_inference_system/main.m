clear;
close all;
clc;

%% Constants

FINAL_SELECTED_FEATURES = 3;
SEQUENTIALFS_HIDDEN_LAYER_SIZE = 18;
HISTOGRAM_BUCKET_SIZE = 0.05;
ACTIVITIES = ["walk", "sit", "run"];

addpath('./fuzzy_inference_system');
addpath('./data_preprocessing');

%% Load dataset and extract the 3 most relevant features

load('../tmp/final_data');

opts = statset('Display', 'iter', 'UseParallel', true);

% Select the most relevant features for the activity target
[fs, ~] = sequentialfs( ...
    @(x, t, hidden_layer_size)sequentialfs_comparison(x, t, SEQUENTIALFS_HIDDEN_LAYER_SIZE) , ...
    final_features_activities_matrix, ...
    final_activities_targets_vector, ...
    'cv', 'none', ...
    'opt', opts, ... 
    'nfeatures', FINAL_SELECTED_FEATURES);

% Prepare the fis dataset
fis_features_activities_matrix = final_features_activities_matrix(:, fs);
fis_activities_targets_vector = final_activities_targets_vector;

save('../tmp/fis_final_data', ...
    'fis_features_activities_matrix', ...
    'fis_activities_targets_vector');

%% Compute features histograms and save histograms

load('../tmp/fis_final_data');

figure(1), histogram(fis_features_activities_matrix(:, 1)', BinWidth=HISTOGRAM_BUCKET_SIZE);
title('Feature 1');

figure(2), histogram(fis_features_activities_matrix(:, 2)', BinWidth=HISTOGRAM_BUCKET_SIZE/2);
title('Feature 2');

figure(3), histogram(fis_features_activities_matrix(:, 3)', BinWidth=HISTOGRAM_BUCKET_SIZE);
title('Feature 3');

saveas(1, '../tmp/fis_feature_1_histogram', 'png');
saveas(2, '../tmp/fis_feature_2_histogram', 'png');
saveas(3, '../tmp/fis_feature_3_histogram', 'png');

%% Find membership functions for input and output

% Feature 1

x = 0:0.01:1;

y1 = gbellmf(x, [0.1 1.3 0.175]);
y2 = gbellmf(x, [0.12 1.3 0.375]);
y3 = gbellmf(x, [0.08 1 0.475]);
y4 = gbellmf(x, [0.12 1.5 0.675]);

figure(4); plot(x, y1, 'black', x, y2, 'red', x, y3, 'green', x, y4, 'blue');
title('Feature 1 Analysis');
xlabel('x');
ylabel('Degree of Membership');

saveas(4, '../tmp/fis_feature_1_membership_function', 'png');

% Feature 2

y1 = gbellmf(x, [0.05 2 0.075]);
y2 = gbellmf(x, [0.12 2 0.475]);
y3 = gbellmf(x, [0.05 1.5 0.675]);
y4 = gbellmf(x, [0.05 2 0.975]);

figure(5); plot(x, y1, 'black', x, y2, 'red', x, y3, 'green', x, y4, 'blue');
title('Feature 2 Analysis');
xlabel('x');
ylabel('Degree of Membership');

saveas(5, '../tmp/fis_feature_2_membership_function', 'png');

% Feature 3

y1 = gbellmf(x, [0.1 1.2 0.175]);
y2 = gbellmf(x, [0.15 1 0.575]);
y3 = gbellmf(x, [0.1 1.2 0.775]);

figure(6); plot(x, y1, 'blue', x, y2, 'red', x, y3, 'green');
title('Feature 3 Analysis');
xlabel('x');
ylabel('Degree of Membership');

saveas(6, '../tmp/fis_feature_3_membership_function', 'png');

% Output
x = 0:0.01:4;

y1 = gbellmf(x, [0.2 2 1]);
y2 = gbellmf(x, [0.2 2 2]);
y3 = gbellmf(x, [0.2 2 3]);

figure(7); plot(x, y1, 'blue', x, y2, 'red', x, y3, 'green');
title('Output Analysis');
xlabel('x');
ylabel('Degree of Membership');

saveas(7, '../tmp/fis_output_membership_function', 'png');

%% Mamdani FIS creation

fis = mamfis('Name' , "MamdaniFis");

fis = addInput(fis, [0 1], 'Name', "feature_1");
fis = addMF(fis, "feature_1", "gbellmf", [0.1 1.3 0.175], 'Name', "low");
fis = addMF(fis, "feature_1", "gbellmf", [0.12 1.3 0.375], 'Name', "medium");
fis = addMF(fis, "feature_1", "gbellmf", [0.08 1 0.475], 'Name', "high");
fis = addMF(fis, "feature_1", "gbellmf", [0.12 1.5 0.675], 'Name', "very high");

fis = addInput(fis, [0 1], 'Name', "feature_2");
fis = addMF(fis, "feature_2", "gbellmf", [0.05 2 0.075], 'Name', "low");
fis = addMF(fis, "feature_2", "gbellmf", [0.12 2 0.475], 'Name', "medium");
fis = addMF(fis, "feature_2", "gbellmf", [0.05 1.5 0.675], 'Name', "high");
fis = addMF(fis, "feature_2", "gbellmf", [0.05 2 0.975], 'Name', "very high");

fis = addInput(fis, [0 1], 'Name', "feature_3");
fis = addMF(fis, "feature_3", "gbellmf", [0.1 1.2 0.175], 'Name', "low");
fis = addMF(fis, "feature_3", "gbellmf", [0.15 1 0.575], 'Name', "medium");
fis = addMF(fis, "feature_3", "gbellmf", [0.1 1.2 0.775], 'Name', "high");

fis = addOutput(fis, [1 3], 'Name', "activity");
fis = addMF(fis, "activity", "gbellmf", [0.2 2 1], 'Name', "sit");
fis = addMF(fis, "activity", "gbellmf", [0.2 2 2], 'Name', "walk");
fis = addMF(fis, "activity", "gbellmf", [0.2 2 3], 'Name', "run");

%% Compute histograms of each feature for each activity

load('../tmp/fis_final_data');

figure(8);

for i = 1 : 3

    fis_features_activity_matrix = fis_features_activities_matrix(fis_activities_targets_vector == i, :);

    for j = 1 : FINAL_SELECTED_FEATURES

        subplot(3, FINAL_SELECTED_FEATURES, (i - 1) * FINAL_SELECTED_FEATURES + j);
        histogram(fis_features_activity_matrix(:, j)', BinWidth=HISTOGRAM_BUCKET_SIZE, BinLimits=[0, 1]);
        title(ACTIVITIES(i) + ": feature " + j);
    end
end

saveas(8, '../tmp/fis_features_of_activities_histograms', 'png');





