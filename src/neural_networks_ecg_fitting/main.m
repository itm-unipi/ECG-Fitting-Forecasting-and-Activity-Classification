clear;
close all;
clc;

%% Load dataset

load('../tmp/final_data');
x = final_features_ecg_mean_matrix';
t = final_ecg_mean_targets_vector';

%% MLP Generation

trainFcn = 'trainbr';   % Bayesian regularization
% trainFcn = 'trainlm'; % Levenberg-Marquardt
% trainFcn = 'trainbfg'; % Quasi-Newton BFGS
% trainFcn = 'trainrp'; % Retropropagazione resiliente
% trainFcn = 'trainscg'; % Gradiente coniugato scalato

hiddenLayerSize = 50;
net = fitnet(hiddenLayerSize,trainFcn);

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 0/100;       % Bayesian regularization does not need validation
net.divideParam.testRatio = 15/100;
net.trainParam.showWindow = 0;

net.performFcn = 'mse';  % Mean Squared Error
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

%% MLP Training and Test

[net,tr] = train(net, x, t);
y = net(x);
figure, plotregression(t, y);