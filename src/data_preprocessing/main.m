clear;
close all;
clc;

%% Constants

N_SUBJECTS = 22;
RESOURCES_PATH = '../resources';
ACTIVITIES = ["walk", "sit", "run"];
WINDOW_SHIFT = 0.2;
WINDOW_SIZE = 50000;

%% Compute Windows number for each signal

min_samples_number = get_min_samples_number(RESOURCES_PATH, ACTIVITIES);

fprintf("min_samples: %d \n", min_samples_number);

windows_number = get_windows_number(min_samples_number, WINDOW_SHIFT, WINDOW_SIZE);

fprintf("windows number: %d \n", windows_number);

%% Generate Features Matrix containing all features of all signals

features_matrix = get_features_matrix(RESOURCES_PATH, WINDOW_SHIFT, WINDOW_SIZE, windows_number);

save('../tmp/non_normalised_features_matrix', 'features_matrix');

%% Normalise and Remove Correlated Features

% TODO


