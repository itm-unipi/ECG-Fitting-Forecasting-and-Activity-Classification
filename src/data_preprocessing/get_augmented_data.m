function [ ...
        augmented_features_matrix, augmented_ecg_mean_targets_vector, augmented_ecg_std_targets_vector ...
    ] = get_augmented_data( ...
        k_fold_window_size, augmentation_factor, features_matrix, ecg_mean_targets_vector, ecg_std_targets_vector ...
    )

    rows_number = size(features_matrix, 1);
    
    % Initalize outputs
    augmented_features_matrix = zeros(size(features_matrix, 1) * augmentation_factor, size(features_matrix, 2));
    augmented_ecg_mean_targets_vector = zeros(size(ecg_mean_targets_vector, 1) * augmentation_factor, 1);
    augmented_ecg_std_targets_vector = zeros(size(ecg_std_targets_vector, 1) * augmentation_factor, 1);
    
    % Copy original dataset in outputs
    augmented_features_matrix(1:rows_number, :) = features_matrix;
    augmented_ecg_mean_targets_vector(1:rows_number, :) = ecg_mean_targets_vector;
    augmented_ecg_std_targets_vector(1:rows_number, :) = ecg_std_targets_vector;
    
    % Generate initial dataset made by features and targets 
    temp_dataset = zeros(size(features_matrix, 1), size(features_matrix, 2) + 2);
    temp_dataset(:, 1:size(features_matrix, 2)) = features_matrix;
    temp_dataset(:, end - 1) = ecg_mean_targets_vector;
    temp_dataset(:, end) = ecg_std_targets_vector;
    
    % Generate a new dataset (same row number of temp_dataset) for every iteration
    for i = 2 : augmentation_factor
    
        % Shuffle the initial dataset
        random_indices = randperm(rows_number);
        shuffled_dataset = temp_dataset(random_indices, :);
    
        % Iterate all k-fold window
        for j = 1 : floor(rows_number / k_fold_window_size)
    
            % Extract training and test set from the suffled dataset
            start_test_index = (j - 1) * k_fold_window_size + 1;
            end_test_index = start_test_index + k_fold_window_size - 1;
            test_dataset = shuffled_dataset(start_test_index : end_test_index, 1 : end - 2);
            training_dataset = shuffled_dataset(:, 1 : end - 2);
            training_dataset(start_test_index : end_test_index, :) = [];
    
            % Train a new autoencoder with the training dataset
            hidden_size = floor(size(training_dataset, 2) / 2);
            autoencoder = trainAutoencoder( ...
                    training_dataset', ...
                    hidden_size,...
                    'EncoderTransferFunction', 'satlin',...
                    'DecoderTransferFunction', 'purelin',...
                    'L2WeightRegularization', 0.01,...
                    'SparsityRegularization', 4,...
                    'SparsityProportion', 0.10, ...
                    'MaxEpochs', 500, ...
                    'ShowProgressWindow', false);

            % Generate new rows using test_dataset
            generated_rows = predict(autoencoder, test_dataset')';
            fprintf('Iteration i: %d, k-fold-step: %d, mse: %f \n', i, j, mse(generated_rows, test_dataset));

            % Copy new rows in output features matrix
            start_result_index = (i - 1) * rows_number + (j - 1) * k_fold_window_size + 1;
            end_result_index = start_result_index + k_fold_window_size - 1;
            augmented_features_matrix(start_result_index : end_result_index, :) = generated_rows; 
        end

        % Copy targets from shuffled dataset in outputs targets vectors
        start_result_index = (i - 1) * rows_number + 1;
        end_result_index = start_result_index + rows_number - 1;
        augmented_ecg_mean_targets_vector(start_result_index : end_result_index, :) = shuffled_dataset(:, end - 1);
        augmented_ecg_std_targets_vector(start_result_index : end_result_index, :) = shuffled_dataset(:, end);
    end
end