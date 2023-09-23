clear;
close all;
clc;

%% Constants

N_SUBJECTS = 22;
RESOURCES_PATH = '../resources';
ACTIVITIES = ["walk", "sit", "run"];
WINDOW_SHIFT = 0.2;
WINDOW_SIZE = 50000;
CORRELATION_THRESHOLD = 0.9;

addpath('./data_preprocessing');

%% Compute Windows number for each signal

min_samples_number = get_min_samples_number(RESOURCES_PATH, ACTIVITIES);

fprintf("min_samples: %d \n", min_samples_number);

windows_number = get_windows_number(min_samples_number, WINDOW_SHIFT, WINDOW_SIZE);

fprintf("windows number: %d \n", windows_number);

%% Generate Features Matrix containing all features of all signals

features_matrix = get_features_matrix(RESOURCES_PATH, WINDOW_SHIFT, WINDOW_SIZE, windows_number);

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

[ecg_mean_targets_vector, ecg_std_targets_vector] = get_ecg_targets_vector(RESOURCES_PATH);

save('../tmp/ecg_targets_vectors', 'ecg_mean_targets_vector', 'ecg_std_targets_vector');

%% Data Augmentation

load('../tmp/uncorrelated_features_matrix');
load('../tmp/ecg_targets_vectors');

k_fold_window_size = 11;
augmentation_factor = 5;

[augmented_features_matrix, augmented_ecg_mean_targets_vector, augmented_ecg_std_targets_vector] = get_augmented_data( ...
    k_fold_window_size, augmentation_factor, uncorrelated_features_matrix, ecg_mean_targets_vector, ecg_std_targets_vector);

save('../tmp/augmented_data', ...
    'augmented_features_matrix', ...
    'augmented_ecg_mean_targets_vector', ...
    'augmented_ecg_std_targets_vector');

%% TODO: normalisation after augmentation
