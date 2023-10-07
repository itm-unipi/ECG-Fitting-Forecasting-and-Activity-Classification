function [dataset, targets] = get_dataset(resources_path)

    % Get names of timeseries files and initialize features matrix
    csv_timeseries = dir(fullfile(resources_path, '*timeseries.csv'));
    csv_targets = dir(fullfile(resources_path, '*targets.csv'));
    dataset = cell(length(csv_timeseries), 1);
    targets = cell(length(csv_targets), 1);

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

        dataset{k} = table2array([ raw_timeseries(1:end - 1, :) raw_targets(1:end - 1, :) ])';
        targets{k} = table2array(raw_targets(2:end, :))';

        fprintf("file: %s\n", csv_timeseries(k).name);
    end
end