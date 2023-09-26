clear;
close all;
clc;

N_REPETION = 5;
MAX_HIDDEN_LAYER_NEURONS = 100;
MIN_HIDDEN_LAYER_NEURONS = 10;
HIDDEN_LAYER_NEURONS_STEP = 5;
results = zeros((MAX_HIDDEN_LAYER_NEURONS - MIN_HIDDEN_LAYER_NEURONS) / HIDDEN_LAYER_NEURONS_STEP, N_REPETION + 1);

%% Load dataset

load('../tmp/final_data');
x = final_features_ecg_mean_matrix';
t = final_ecg_mean_targets_vector';

trainFcn = 'trainbr';   % Bayesian regularization
% trainFcn = 'trainlm'; % Levenberg-Marquardt
% trainFcn = 'trainbfg'; % Quasi-Newton BFGS
% trainFcn = 'trainrp'; % Retropropagazione resiliente
% trainFcn = 'trainscg'; % Gradiente coniugato scalato

i = 1;
while true
    hidden_layer_size = MIN_HIDDEN_LAYER_NEURONS + (i - 1) * HIDDEN_LAYER_NEURONS_STEP;
    
    if hidden_layer_size > MAX_HIDDEN_LAYER_NEURONS
            break;
    end

    results(i, 1) = hidden_layer_size;
    for j = 1 : N_REPETION
        net = fitnet(hidden_layer_size,trainFcn);
        
        net.divideFcn = 'dividerand';  % Divide data randomly
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 85/100;
        net.divideParam.valRatio = 0/100;       % Bayesian regularization does not need validation
        net.divideParam.testRatio = 15/100;
        net.trainParam.showWindow = 0;
        
        net.performFcn = 'mse';  % Mean Squared Error
        net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
            'plotregression', 'plotfit'};
        
        [net,tr] = train(net, x, t);
        y = net(x);
        % figure, plotregression(t, y);
        regression_stats = fitlm(t',y');
        r_value = sqrt(regression_stats.Rsquared.Ordinary);
        results(i, j + 1) = r_value;
        fprintf("hidden neurons: %d, repetition: %d, r-value: %d\n", hidden_layer_size, j, r_value);
    end
    i = i + 1;
end

save('../tmp/mlp_ecg_fitting_results', results);