function features_matrix = get_features_matrix(resources_path, window_shift, window_size, windows_number)

    FEATURES_NUMBER = 13;
    SIGNALS_NUMBER = 11;

    % Get names of timeseries files and initialize features matrix
    csv_timeseries = dir(fullfile(resources_path, '*timeseries.csv'));
    features_matrix = zeros(length(csv_timeseries), FEATURES_NUMBER * windows_number * SIGNALS_NUMBER);

    % Iterate timeseries files
    for k = 1 : length(csv_timeseries)
        
        % Read timeseries files and remove timestamp column
        file_path = fullfile(resources_path, csv_timeseries(k).name);
        raw_data = readtable(file_path);
        raw_data = raw_data(:, 2:end);
        
        % Iterate all the signals (raw data columns)
        for i = 1 : size(raw_data, 2)

            % Iterate windows of a signal
            for j = 1 : windows_number
                
                fprintf("file: %d, signal: %d, window: %d\n", k, i, j);
                
                % Extract raw data from window
                start_row_index = (j - 1) * floor(window_shift * window_size) + 1;
                end_row_index = start_row_index + window_size - 1;
                window_raw_data = table2array(raw_data(start_row_index : end_row_index, i));
                
                % Compute column indices of features matrix in witch copy
                % the computed features from the window
                start_features_column_index = ((i - 1) * windows_number + (j - 1)) * FEATURES_NUMBER + 1; 
                end_features_column_index = start_features_column_index + FEATURES_NUMBER - 1;

                % Compute and insert features in the features matrix
                features_matrix(k, start_features_column_index : end_features_column_index) =  [
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
            end
        end
    end
end

