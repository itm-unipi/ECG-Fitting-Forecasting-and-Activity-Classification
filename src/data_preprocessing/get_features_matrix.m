function [features_matrix, activities_targets_vector] = get_features_matrix(resources_path, window_shift, window_size, activities)

    FEATURES_NUMBER = 13;
    SIGNALS_NUMBER = 11;

    % Get names of timeseries files and initialize features matrix
    csv_timeseries = dir(fullfile(resources_path, '*timeseries.csv'));
    features_matrix = [];
    activities_targets_vector = [];

    % Iterate timeseries files
    for k = 1 : length(csv_timeseries)
        
        % Read timeseries files and remove timestamp column
        file_path = fullfile(resources_path, csv_timeseries(k).name);
        raw_data = readtable(file_path);
        raw_data = raw_data(:, 2:end);
        
        % Extract the activity of the file
        for i = 1 : size(activities, 2)
            if contains(csv_timeseries(k).name, activities(i))
                activity = i;
                break;
            end
        end
        
        % Iterate all possible windows
        i = 1;
        while true

            % Compute window row indices
            start_window_index = (i - 1) * floor(window_shift * window_size) + 1;
            end_window_index = start_window_index + window_size - 1;

            % Check if the window exceeds the raw_data rows number 
            if(end_window_index > size(raw_data, 1))
                break;
            end

            % Initialize the new window features row
            window_features = zeros(1, SIGNALS_NUMBER * FEATURES_NUMBER);
            
            % Iterate all the signals (raw data columns)
            for j = 1 : size(raw_data, 2)
                
                fprintf("file: %d, window: %d, signal: %d\n", k, i, j);
                
                % Extract raw data from window and compute features
                window_raw_data = table2array(raw_data(start_window_index : end_window_index, j));
                signal_features = [
                    mean(window_raw_data),      ...     % Mean
                    median(window_raw_data),    ...     % Median
                    var(window_raw_data),       ...     % Variance
                    std(window_raw_data),       ...     % Standard Deviation
                    min(window_raw_data),       ...     % Minimum
                    max(window_raw_data),       ...     % Maximum
                    kurtosis(window_raw_data),  ...     % Kurtosis
                    skewness(window_raw_data),  ...     % Skewness
                    iqr(window_raw_data),       ...     % Interqauntile Range Gsr
                    sum(window_raw_data .^ 2),  ...     % Energy
                    meanfreq(window_raw_data),  ...     % Mean Frequency
                    medfreq(window_raw_data),   ...     % Median Frequency
                    obw(window_raw_data)                % Occupied Bandwidth
                ];
    
                % Copy the new computed values in the window features row
                start_signal_index = (j - 1) * FEATURES_NUMBER + 1; 
                end_signal_index = start_signal_index + FEATURES_NUMBER - 1;
                window_features(1, start_signal_index : end_signal_index) = signal_features;
            end

            % Concatenate the new window features row with the features_matrix and 
            % the activity of the window with the activity vector
            features_matrix = [features_matrix; window_features];
            activities_targets_vector = [activities_targets_vector; activity];
            
            i = i + 1;
        end
    end
end

