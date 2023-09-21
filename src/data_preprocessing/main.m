clear;
close all;
clc;

%% constants and variables

% dataset directory
DATASET_FOLDER = 'dataset';
% feature's matrix dimension
NUMBER_OF_SIGNALS = 11;
NUMBER_OF_FEATURES = 11;
NUMBER_OF_TIMESERIES = length(dir(fullfile(DATASET_FOLDER,'*timeseries.csv')));
NUMBER_OF_TARGETS = length(dir(fullfile(DATASET_FOLDER,'*targets.csv')));
M_FEATURES_ROWS = NUMBER_OF_TIMESERIES;
M_FEATURES_COLUMNS = NUMBER_OF_SIGNALS * NUMBER_OF_FEATURES;
TIMESERIES_WINDOW_SIZE = ;

% data matrices
m_features = zeros(M_FEATURES_ROWS, M_FEATURES_COLUMNS);
m_features_contiguous_windows = zeros(M_FEATURES_ROWS, M_FEATURES_COLUMNS * c.windows_number_contiguos);
m_features_overlapped_windows = zeros(M_FEATURES_ROWS, M_FEATURES_COLUMNS * c.windows_number_overlapped);
m_mean_ecg = zeros(M_FEATURES_ROWS, 1);
m_std_ecg = zeros(M_FEATURES_ROWS, 1);
m_activies = zeros(M_FEATURES_ROWS, 1);

%% biajo