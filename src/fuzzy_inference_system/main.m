clear;
close all;
clc;

%% Constants

FINAL_SELECTED_FEATURES = 4;
SEQUENTIALFS_HIDDEN_LAYER_SIZE = 18;
HISTOGRAM_BUCKET_SIZE = 0.05;
ACTIVITIES = ["walk", "sit", "run"];

addpath('./fuzzy_inference_system');
addpath('./data_preprocessing');
rng("default");
figure_id = 1;

%% Load dataset and extract the 3 most relevant features

load('../tmp/final_data');

%{
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
%}

filter = [false false true false false true false false false false true false true false];
fis_features_activities_matrix = final_features_activities_matrix(:, filter);
fis_activities_targets_vector = final_activities_targets_vector;

save('../tmp/fis_final_data', ...
    'fis_features_activities_matrix', ...
    'fis_activities_targets_vector');

%% Compute features histograms and save histograms

load('../tmp/fis_final_data');

for i = 1 : FINAL_SELECTED_FEATURES
    
    figure(figure_id);
    histogram(fis_features_activities_matrix(:, i)', BinWidth=HISTOGRAM_BUCKET_SIZE);
    title("Feature " + i);

    saveas(figure_id, "../tmp/fis_feature_" + i + "_histogram", 'png');
    figure_id = figure_id + 1;
end

%% Find membership functions for input and output

% Feature 1

x = 0:0.01:1;

y1 = gbellmf(x, [0.03 2 0.025]);
y2 = gbellmf(x, [0.1 1.5 0.4]);
y3 = gbellmf(x, [0.2 2.5 0.725]);

figure(figure_id); plot(x, y1, 'blue', x, y2, 'red', x, y3, 'green');
title('Feature 1 Analysis');
xlabel('x');
ylabel('Degree of Membership');

saveas(figure_id, '../tmp/fis_feature_1_membership_function', 'png');
figure_id = figure_id + 1;

% Feature 2

y1 = gbellmf(x, [0.18 1.5 0.01]);
y2 = gbellmf(x, [0.2 1.5 0.275]);
y3 = gbellmf(x, [0.1 1.5 0.525]);
y4 = gbellmf(x, [0.1 1.5 0.825]);

figure(figure_id); plot(x, y1, 'black', x, y2, 'red', x, y3, 'green', x, y4, 'blue');
title('Feature 2 Analysis');
xlabel('x');
ylabel('Degree of Membership');

saveas(figure_id, '../tmp/fis_feature_2_membership_function', 'png');
figure_id = figure_id + 1;

% Feature 3 

y1 = gbellmf(x, [0.06 1.5 0.125]);
y2 = gbellmf(x, [0.03 1.5 0.475]);
y3 = gbellmf(x, [0.02 1.8 0.675]);

figure(figure_id); plot(x, y1, 'blue', x, y2, 'red', x, y3, 'green');
title('Feature 3 Analysis');
xlabel('x');
ylabel('Degree of Membership');

saveas(figure_id, '../tmp/fis_feature_3_membership_function', 'png');
figure_id = figure_id + 1;

% Feature 4

y1 = gbellmf(x, [0.05 2 0.1]);
y2 = gbellmf(x, [0.145 2 0.475]);
y3 = gbellmf(x, [0.04 2 0.95]);

figure(figure_id); plot(x, y1, 'blue', x, y2, 'red', x, y3, 'green');
title('Feature 4 Analysis');
xlabel('x');
ylabel('Degree of Membership');

saveas(figure_id, '../tmp/fis_feature_3_membership_function', 'png');
figure_id = figure_id + 1;

% Output
x = 0:0.01:4;

y1 = gbellmf(x, [0.2 2 1]);
y2 = gbellmf(x, [0.2 2 2]);
y3 = gbellmf(x, [0.2 2 3]);

figure(figure_id); plot(x, y1, 'blue', x, y2, 'red', x, y3, 'green');
title('Output Analysis');
xlabel('x');
ylabel('Degree of Membership');

saveas(figure_id, '../tmp/fis_output_membership_function', 'png');
figure_id = figure_id + 1;

%% Mamdani FIS creation

fis = mamfis('Name' , "MamdaniFis");

fis = addInput(fis, [0 1], 'Name', "f1");
fis = addMF(fis, "f1", "gbellmf", [0.03 2 0.025], 'Name', "low");
fis = addMF(fis, "f1", "gbellmf", [0.1 1.5 0.4], 'Name', "medium");
fis = addMF(fis, "f1", "gbellmf", [0.2 2.5 0.725], 'Name', "high");

fis = addInput(fis, [0 1], 'Name', "f2");
fis = addMF(fis, "f2", "gbellmf", [0.18 1.5 0.01], 'Name', "low");
fis = addMF(fis, "f2", "gbellmf", [0.2 1.5 0.275], 'Name', "medium");
fis = addMF(fis, "f2", "gbellmf", [0.1 1.5 0.525], 'Name', "high");
fis = addMF(fis, "f2", "gbellmf", [0.1 1.5 0.825], 'Name', "very_high");

fis = addInput(fis, [0 1], 'Name', "f3");
fis = addMF(fis, "f3", "gbellmf", [0.06 1.5 0.125], 'Name', "low");
fis = addMF(fis, "f3", "gbellmf", [0.03 1.5 0.475], 'Name', "medium");
fis = addMF(fis, "f3", "gbellmf", [0.02 1.8 0.675], 'Name', "high");

fis = addInput(fis, [0 1], 'Name', "f4");
fis = addMF(fis, "f4", "gbellmf", [0.05 2 0.1], 'Name', "low");
fis = addMF(fis, "f4", "gbellmf", [0.145 2 0.475], 'Name', "medium");
fis = addMF(fis, "f4", "gbellmf", [0.04 2 0.95], 'Name', "high");

fis = addOutput(fis, [1 3], 'Name', "activity");
fis = addMF(fis, "activity", "gbellmf", [0.2 2 1], 'Name', "sit");
fis = addMF(fis, "activity", "gbellmf", [0.2 2 2], 'Name', "walk");
fis = addMF(fis, "activity", "gbellmf", [0.2 2 3], 'Name', "run");

% Generate rules using Wang-Mendel method
rules = get_rules(fis, fis_features_activities_matrix, fis_activities_targets_vector);
fis = addRule(fis, rules);

%% Test the FIS

%fis.DefuzzificationMethod = "lom";

y = evalfis(fis, fis_features_activities_matrix);
error = mse(y, fis_activities_targets_vector);
