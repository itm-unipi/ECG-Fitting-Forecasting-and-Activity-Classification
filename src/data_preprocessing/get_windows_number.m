function windows_number = get_windows_number(min_samples_number, window_shift, window_size)

    i = 0;

    while true
        
        % Compute fist and last index of the actual window
        start_index = i * floor(window_shift * window_size);
        end_index = start_index + window_size;

        % Check if the actual window go out of range
        if end_index > min_samples_number
            break;
        end

        i = i + 1;
    end

    windows_number = i;
end