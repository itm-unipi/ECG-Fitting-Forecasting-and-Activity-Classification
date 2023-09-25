function perf = sequentialfs_comparison(x, t) 
    
    % Create a network 
    hiddenLayerSize = 3; 
    net = fitnet(hiddenLayerSize); 
    net.trainParam.showWindow = 0;

    % Train the network 
    [net, ~] = train(net, x', t'); 
     
    % Test the network 
    y = net(x'); 
    perf = perform(net, t', y); 
end