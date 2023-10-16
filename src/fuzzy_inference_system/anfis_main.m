clear;
close all;
clc;

%% Constants

N_MEMBERSHIP_FUNCTIONS = 3;
INPUT_MEMBERSHIP_FUNCTION_TYPE = 'gbellmf';
OUTPUT_MEMBERSHIP_FUNCTION_TYPE = 'linear';
FRACTION_TEST_SET = 0.15;
N_EPOCHS = 10;

rng("default");

%% Load and prepare dataset 

load('../tmp/fis_final_data');

% Partition the dataset into two equal set for training and checking
dataset = [fis_features_activities_matrix fis_activities_targets_vector];
partition_data = cvpartition(size(fis_features_activities_matrix, 1), "Holdout", FRACTION_TEST_SET);

training_data = dataset(1:2:size(dataset, 1), :);
checking_data = dataset(2:2:size(dataset, 1), :);

%% Create the ANFIS

options = genfisOptions('GridPartition');
options.NumMembershipFunctions = N_MEMBERSHIP_FUNCTIONS;
options.InputMembershipFunctionType = INPUT_MEMBERSHIP_FUNCTION_TYPE;
options.OutputMembershipFunctionType = OUTPUT_MEMBERSHIP_FUNCTION_TYPE;
fismat = genfis(training_data(:, 1:5), training_data(:, 6), options);

%% Train the ANFIS

[net, training_error, ~, ~, checking_error] = anfis(training_data, fismat, N_EPOCHS, [], checking_data);

%% Test the ANFIS

% Predict the output
y = evalfis(net, checking_data(:, 1:end-1));

% Round of the predicted value to the neareast integer value
y_rounded = round(y);

for i = 1 : size(y_rounded, 1)
    if y_rounded(i) < 1
        y_rounded(i) = 1;
    elseif y_rounded(i) > 3
        y_rounded(i) = 3;
    end
end

% Encode output and target to generate the confusion matrix
encoded_y = full(ind2vec(y_rounded'));
encoded_t = full(ind2vec(checking_data(:, end)'));

figure(1);
plotconfusion(encoded_t, encoded_y);
saveas(1, '../tmp/anfis_confusion_matrix', 'png');

% Evaluate correct classification percentage
[c, ~] = confusion(encoded_t, encoded_y);
correct_classification_percentage = 100 * (1 - c);
fprintf("Number of membership functions: %d, Input Membership function type: %s, Output Membership function type: %s, Epochs: %d, Correct classification: %d%%\n", N_MEMBERSHIP_FUNCTIONS, INPUT_MEMBERSHIP_FUNCTION_TYPE, OUTPUT_MEMBERSHIP_FUNCTION_TYPE, N_EPOCHS, correct_classification_percentage);