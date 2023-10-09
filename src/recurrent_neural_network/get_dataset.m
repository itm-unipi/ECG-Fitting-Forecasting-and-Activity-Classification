function [dataset, targets] = get_dataset(resources_path, window_size)

    % Get names of timeseries files and initialize features matrix
    csv_timeseries = dir(fullfile(resources_path, '*timeseries.csv'));
    csv_targets = dir(fullfile(resources_path, '*targets.csv'));
    dataset = {};
    targets = {};

    % Iterate timeseries files
    for k = 1 : length(csv_timeseries) 

        % Read timeseries files and remove timestamp column
        file_path = fullfile(resources_path, csv_timeseries(k).name);
        raw_timeseries = readtable(file_path);
        raw_timeseries = raw_timeseries(:, 2:end);

        % Read targets files and remove timestamp column
        file_path = fullfile(resources_path, csv_targets(k).name);
        raw_targets = readtable(file_path);
        raw_targets = raw_targets(:, 2);

        % Iterate all possible windows
        i = 1;
        while true

            % Compute window row indices
            start_window_index = (i - 1) * window_size + 1;
            end_window_index = start_window_index + window_size - 1;

            % Check if the window exceeds the raw_data rows number 
            if(end_window_index > size(raw_timeseries, 1))
                break;
            end

            dataset = [ 
                dataset; 
                { table2array([ raw_timeseries(start_window_index : end_window_index - 1, :) raw_targets(start_window_index : end_window_index - 1, :) ])' }
            ];
            targets = [ 
                targets; 
                { table2array(raw_targets(start_window_index + 1 : end_window_index, :))' }
            ];
    
            fprintf("file: %s, window: %d\n", csv_timeseries(k).name, i);

            i = i + 1;
        end
    end
end