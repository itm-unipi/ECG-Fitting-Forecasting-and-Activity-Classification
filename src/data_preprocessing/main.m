clear;
close all;
clc;

%% Constants

N_SUBJECTS = 22;
RESOURCES_PATH = '../resources';
ACTIVITIES = ["walk", "sit", "run"];
WINDOW_SHIFT = 0.3;
WINDOW_SIZE = 50000;
CORRELATION_THRESHOLD = 0.9;
K_FOLD_WINDOW_SIZE = 111;
AUGMENTATION_FACTOR = 3;
FINAL_SELECTED_FEATURES = 10;
SEQUENTIALFS_HIDDEN_LAYER_SIZE = 18;

addpath('./data_preprocessing');

%% Generate Features Matrix containing all features of all signals

features_matrix = get_features_matrix(RESOURCES_PATH, WINDOW_SHIFT, WINDOW_SIZE);

save('../tmp/non_normalised_features_matrix', 'features_matrix');

%% Normalise and Remove Correlated Features

load('../tmp/non_normalised_features_matrix');

% Normalise features matrix
non_negative_features_matrix = features_matrix - min(features_matrix);
normalized_features_matrix = non_negative_features_matrix ./ max(non_negative_features_matrix);

% Remove correlated features
correlation_matrix = corrcoef(normalized_features_matrix);
[correlated_columns_indices, ~] = find(tril((abs(correlation_matrix) > CORRELATION_THRESHOLD), -1));
correlated_columns_indices = unique(sort(correlated_columns_indices));

uncorrelated_features_matrix = normalized_features_matrix;
uncorrelated_features_matrix(:, correlated_columns_indices) = [];

save('../tmp/uncorrelated_features_matrix', 'uncorrelated_features_matrix');

%% Get ECG Mean and Standard Deviation Vectors (Targets)

[ecg_mean_targets_vector, ecg_std_targets_vector] = get_ecg_targets_vector(RESOURCES_PATH, WINDOW_SHIFT, WINDOW_SIZE);

save('../tmp/ecg_targets_vectors', 'ecg_mean_targets_vector', 'ecg_std_targets_vector');

%% Data Augmentation

load('../tmp/uncorrelated_features_matrix');
load('../tmp/ecg_targets_vectors');

[augmented_features_matrix, augmented_ecg_mean_targets_vector, augmented_ecg_std_targets_vector] = get_augmented_data( ...
    K_FOLD_WINDOW_SIZE, AUGMENTATION_FACTOR, uncorrelated_features_matrix, ecg_mean_targets_vector, ecg_std_targets_vector);

% Normalise augmented features matrix
non_negative_augmented_features_matrix = augmented_features_matrix - min(augmented_features_matrix);
normalized_augmented_features_matrix = non_negative_augmented_features_matrix ./ max(non_negative_augmented_features_matrix);

save('../tmp/augmented_data', ...
    'normalized_augmented_features_matrix', ...
    'augmented_ecg_mean_targets_vector', ...
    'augmented_ecg_std_targets_vector');

%% Extraction of 10 Best Features

load('../tmp/augmented_data');

opts = statset('Display', 'iter', 'UseParallel', true);

% Select the most relevant features for the mean ecg targets
[fs_mean, ~] = sequentialfs( ...
    @(x, t, hidden_layer_size)sequentialfs_comparison(x, t, SEQUENTIALFS_HIDDEN_LAYER_SIZE) , ...
    normalized_augmented_features_matrix, ...
    augmented_ecg_mean_targets_vector, ...
    'cv', 'none', ...
    'opt', opts, ... 
    'nfeatures', FINAL_SELECTED_FEATURES);

% Select the most relevant features for the standard deviation ecg targets
[fs_std, ~] = sequentialfs( ...
     @(x, t, hidden_layer_size)sequentialfs_comparison(x, t, SEQUENTIALFS_HIDDEN_LAYER_SIZE) , ...
    normalized_augmented_features_matrix, ...
    augmented_ecg_std_targets_vector, ...
    'cv', 'none', ...
    'opt', opts, ... 
    'nfeatures', FINAL_SELECTED_FEATURES);

% Prepare the final dataset
final_features_ecg_mean_matrix = normalized_augmented_features_matrix(:, fs_mean);
final_features_ecg_std_matrix = normalized_augmented_features_matrix(:, fs_std);
final_ecg_mean_targets_vector = augmented_ecg_mean_targets_vector;
final_ecg_std_targets_vector = augmented_ecg_std_targets_vector;

save('../tmp/final_data', ...
    'final_features_ecg_mean_matrix', ...
    'final_features_ecg_std_matrix', ...
    'final_ecg_mean_targets_vector', ...
    'final_ecg_std_targets_vector');

