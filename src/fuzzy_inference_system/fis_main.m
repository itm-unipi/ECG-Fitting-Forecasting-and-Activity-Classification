clear;
close all;
clc;

%% Constants

FINAL_SELECTED_FEATURES = 5;
SEQUENTIALFS_HIDDEN_LAYER_SIZE = 18;
N_TRIANGLES = 30;
DEFUZZIFICATION_METHOD = "mom";

addpath('./fuzzy_inference_system');
addpath('./data_preprocessing');
rng("default");
figure_id = 1;

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

%% Find membership functions for input and output

x = 0:0.0001:1;
step = 1/(N_TRIANGLES + 1);

% Iterate all features
for i = 1 : FINAL_SELECTED_FEATURES

    step = 1 / (N_TRIANGLES - 1);

    figure(figure_id);
    title("Feature " + i + " Analysis");
    xlabel('x');
    ylabel('Degree of Membership');
    hold on;
    
    % Create all triangular membership functions
    for j = 1 : N_TRIANGLES

        center = (j - 1) * step;
        membership_function = trimf(x, [(center - step) center (center + step)]);
        plot(x, membership_function);
    end

    saveas(figure_id, "../tmp/fis_feature_" + i + "_membership_function", 'png');
    figure_id = figure_id + 1;
end

x = 0:0.0001:4;

figure(figure_id)
title("Output Analysis");
xlabel('x');
ylabel('Degree of Membership');
hold on;

% Create output triangular membership function
membership_function = trimf(x, [0 1 2]);
plot(x, membership_function);

membership_function = trimf(x, [1 2 3]);
plot(x, membership_function);

membership_function = trimf(x, [2 3 4]);
plot(x, membership_function);

saveas(figure_id, '../tmp/fis_output_membership_function', 'png');
figure_id = figure_id + 1;


%% Mamdani FIS creation

fis = mamfis('Name' , "MamdaniFis");

% Add input for every feature 
for i = 1 : FINAL_SELECTED_FEATURES
    
    fis = addInput(fis, [0 1], 'Name', "f" + i);
    step = 1 / (N_TRIANGLES + 1);
    
    % Add the triangular membership functions for every input
    for j = 1 : N_TRIANGLES

        center = j * step;
        fis = addMF(fis, "f" + i, "trimf", [(center - step) center (center + step)], 'Name', "FS" + j);
    end
end

% Add the output and its membership functions
fis = addOutput(fis, [1 3], 'Name', "activity");
fis = addMF(fis, "activity", "trimf", [0 1 2], 'Name', "sit");
fis = addMF(fis, "activity", "trimf", [1 2 3], 'Name', "walk");
fis = addMF(fis, "activity", "trimf", [2 3 4], 'Name', "run");

% Generate rules using Wang-Mendel method
rules = get_rules(fis, fis_features_activities_matrix, fis_activities_targets_vector);
fis = addRule(fis, rules);

%% Test the FIS

fis.DefuzzificationMethod = DEFUZZIFICATION_METHOD;
    
% Predict the output and compute the error
y = evalfis(fis, fis_features_activities_matrix);
error = mse(y, fis_activities_targets_vector);

% Encode output and target to generate the confusion matrix
encoded_y = full(ind2vec(round(y)'));
encoded_t = full(ind2vec(fis_activities_targets_vector'));

figure(figure_id);
plotconfusion(encoded_t, encoded_y);
saveas(figure_id, '../tmp/fis_confusion_matrix', 'png');
figure_id = figure_id + 1;

% Evaluate correct classification percentage
[c, ~] = confusion(encoded_t, encoded_y);
correct_classification_percentage = 100 * (1 - c);
fprintf("Number Triangles: %d, Defuzzification Method: %s, MSE: %d, Correct classification: %d%%\n", N_TRIANGLES, DEFUZZIFICATION_METHOD, error, correct_classification_percentage);
