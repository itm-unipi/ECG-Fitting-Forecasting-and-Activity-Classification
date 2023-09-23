function [ecg_mean_targets_vector, ecg_std_targets_vector] = get_ecg_targets_vector(resources_path)

    % Get names of targets files
    csv_targets = dir(fullfile(resources_path, '*targets.csv'));
    samples_number = length(csv_targets);
    
    % Initialize targets vectors
    ecg_mean_targets_vector = zeros(samples_number, 1);
    ecg_std_targets_vector = zeros(samples_number, 1);
    
    % Iterate all targets file
    for k = 1 : samples_number
        % Read targets files
        file_path = fullfile(resources_path, csv_targets(k).name);
        raw_data = readtable(file_path);

        % Compute mean and standard deviation
        ecg_mean = mean(table2array(raw_data(:,2)));
        ecg_std = std(table2array(raw_data(:,2)));
        ecg_mean_targets_vector(k) = ecg_mean;
        ecg_std_targets_vector(k) = ecg_std;

        fprintf("file: %s, mean: %d, std: %d \n", csv_targets(k).name, ecg_mean, ecg_std);
    end
end