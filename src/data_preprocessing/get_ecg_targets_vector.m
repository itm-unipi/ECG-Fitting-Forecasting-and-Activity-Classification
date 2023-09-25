function [ecg_mean_targets_vector, ecg_std_targets_vector] = get_ecg_targets_vector(resources_path, window_shift, window_size)

    % Get names of targets files
    csv_targets = dir(fullfile(resources_path, '*targets.csv'));
    samples_number = length(csv_targets);
    
    % Initialize targets vectors
    ecg_mean_targets_vector = [];
    ecg_std_targets_vector = [];
    
    % Iterate all targets file
    for k = 1 : samples_number

        % Read targets files
        file_path = fullfile(resources_path, csv_targets(k).name);
        raw_data = readtable(file_path);

        % Iterate all windows
        i = 1;
        while true

            % Compute window row indices
            start_window_index = (i - 1) * floor(window_shift * window_size) + 1;
            end_window_index = start_window_index + window_size - 1;

            % Check if the window exceeds the raw_data rows number 
            if(end_window_index > size(raw_data, 1))
                break;
            end
            
            % Compute mean and standard deviation of the window
            ecg_mean = mean(table2array(raw_data(start_window_index : end_window_index, 2)));
            ecg_std = std(table2array(raw_data(start_window_index : end_window_index, 2)));
            ecg_mean_targets_vector = [ecg_mean_targets_vector; ecg_mean]; 
            ecg_std_targets_vector = [ecg_std_targets_vector; ecg_std];

            fprintf("file: %s, window: %d, mean: %d, std: %d \n", csv_targets(k).name, i, ecg_mean, ecg_std);

            i = i + 1;
        end
    end
end