clear;
close all;
clc;

%% Constants

FINAL_SELECTED_FEATURES = 5;
SEQUENTIALFS_HIDDEN_LAYER_SIZE = 18;
MIN_TRIANGLES = 5;
MAX_TRIANGLES = 50;
TRIANGLES_STEP = 5;
DEFUZZIFICATION_METHODS = ["centroid", "bisector", "lom", "som", "mom"];

addpath('./fuzzy_inference_system');
addpath('./data_preprocessing');
rng("default");

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

%% Test different triangles number

results = zeros(floor((MAX_TRIANGLES - MIN_TRIANGLES) / TRIANGLES_STEP), size(DEFUZZIFICATION_METHODS, 2) + 1);
k = 1;

while true  

    % Compute the new triangles number
    n_triangles = MIN_TRIANGLES + (k - 1) * TRIANGLES_STEP;
    
    % Check if the triangles number has reached the max value
    if n_triangles > MAX_TRIANGLES
        break;
    end

    % Mamdani FIS creation
    fis = mamfis('Name' , "MamdaniFis");
    
    % Add input for every feature 
    for i = 1 : FINAL_SELECTED_FEATURES
        
        fis = addInput(fis, [0 1], 'Name', "f" + i);
        step = 1 / (n_triangles + 1);
        
        % Add the triangular membership functions for every input
        for j = 1 : n_triangles
    
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

    % Test the network with different defuzzification methods
    for i = 1: size(DEFUZZIFICATION_METHODS, 2)
    
        fis.DefuzzificationMethod = DEFUZZIFICATION_METHODS(i);
        
        % Predict the output
        y = evalfis(fis, fis_features_activities_matrix);
        
        % Encode output and target to generate the confusion matrix
        encoded_y = full(ind2vec(round(y)'));
        encoded_t = full(ind2vec(fis_activities_targets_vector'));
    
        % Evaluate and save correct classification percentage
        [c, ~] = confusion(encoded_t, encoded_y);
        correct_classification_percentage = 100 * (1 - c);
        fprintf("Number Triangles: %d, Defuzzification Method: %s, Correct classification: %d%%\n", n_triangles, DEFUZZIFICATION_METHODS(i), correct_classification_percentage);
        results(k, i + 1) = correct_classification_percentage;
    end
    
    results(k, 1) = n_triangles;
    k = k + 1;
end