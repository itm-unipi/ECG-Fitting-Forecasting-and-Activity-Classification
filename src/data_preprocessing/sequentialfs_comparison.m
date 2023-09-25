function perf = sequentialfs_comparison(x, t, hidden_layer_size) 

    % Create a network 
    net = fitnet(hidden_layer_size); 
    net.trainParam.showWindow = 0;
    net.performFcn = 'mse';

    % Train the network 
    [net, ~] = train(net, x', t'); 
     
    % Test the network 
    y = net(x'); 
    perf = perform(net, t', y); 
end