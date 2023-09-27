clear;
close all;
clc;

%% Constants

ACTIVITIES = ["walk", "sit", "run"];
N_REPETITION = 5;
MAX_HIDDEN_LAYER_NEURONS = 100;
MIN_HIDDEN_LAYER_NEURONS = 15;
HIDDEN_LAYER_NEURONS_STEP = 5;

%% Load Data and Initialize variables

load('../tmp/final_data');

x = final_features_activities_matrix';
t = full(ind2vec(final_activities_targets_vector'));

results = zeros(floor((MAX_HIDDEN_LAYER_NEURONS - MIN_HIDDEN_LAYER_NEURONS) / HIDDEN_LAYER_NEURONS_STEP), N_REPETITION + 1);

%% MLP Training and Test

i = 1;
while true

    % Compute the new hidden layer size
    hidden_layer_size = MIN_HIDDEN_LAYER_NEURONS + (i - 1) * HIDDEN_LAYER_NEURONS_STEP;
    
    % Check if hidden_layer_size has reached the max value
    if hidden_layer_size > MAX_HIDDEN_LAYER_NEURONS
        break;
    end

    % Save the new hidden layer size
    results(i, 1) = hidden_layer_size;

    % Iterate all repetitions
    for j = 1 : N_REPETITION
        
        % Create and train a mlp network
        net = patternnet(hidden_layer_size);
        net.trainParam.showWindow = 0;
        [net, tr] = train(net, x, t);

        % Test the network and save results
        test_x = x(:, tr.testInd);
        test_t = t(:, tr.testInd);
        test_y = net(test_x);
        [c, ~] = confusion(test_t, test_y);
        correct_classification_percentage = 100 * (1 - c);
        results(i, j + 1) = correct_classification_percentage;

        fprintf("hidden neurons: %d, repetition: %d, correct classification: %d%%\n", hidden_layer_size, j, correct_classification_percentage);
    end

    i = i + 1;
end

writematrix(results, fullfile('../tmp', 'mlp_activities_classification_results.csv'));
save('../tmp/mlp_activities_classification_results', 'results');
