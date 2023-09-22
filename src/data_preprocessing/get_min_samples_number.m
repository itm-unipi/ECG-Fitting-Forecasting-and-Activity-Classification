function min_samples_number = get_min_samples_number(resources_path, activities)

    csv_files = dir(fullfile(resources_path, '*.csv'));
    min_samples_number = 0;
    
    for i = 1 : length(csv_files) / 6
        for j = 1 : length(activities)
            
            % Generate filename
            filename = strcat("s", num2str(i), "_", activities(j), "_timeseries.csv");
            file_path = fullfile(resources_path, filename);
            
            % Extract number of samples from the file
            raw_data = readtable(file_path);
            n_samples = size(raw_data, 1);

            % Update min_sample_number
            if min_samples_number == 0 || min_samples_number > n_samples
                min_samples_number = n_samples;
            end

            fprintf("file: %s, samples: %d \n", filename, n_samples); % TEST
        end
    end
end