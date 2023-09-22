clear;
close all;
clc;

%% Constants

N_SUBJECTS = 22;
RESOURCES_PATH = '../resources';
ACTIVITIES = ["walk", "sit", "run"];
WINDOW_SHIFT = 0.2;
WINDOW_SIZE = 10;

%% Load row data from dataset

csv_files = dir(fullfile(RESOURCES_PATH, '*.csv'));
min_samples_number = get_min_samples_number(RESOURCES_PATH, csv_files, ACTIVITIES);

fprintf("min_samples: %d \n", min_samples_number);

windows_number = get_windows_number(min_samples_number, WINDOW_SHIFT, WINDOW_SIZE);

fprintf("windows number: %d \n", windows_number);

%% feature extraction

