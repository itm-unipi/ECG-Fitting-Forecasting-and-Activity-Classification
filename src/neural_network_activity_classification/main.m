% Create and train a mlp network
net = patternnet(10);
% net.trainParam.showWindow = 0;
[net, tr] = train(net, x, t);

plotperform(tr);

testX = x(:, tr.testInd);
testT = t(:, tr.testInd);

testY = net(testX);
testIndices = vec2ind(testY);

plotconfusion(testT, testY);

[c, cm] = confusion(testT, testY);
fprintf('Percentage Correct Classification: %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification: %f%%\n', 100*c);

figure;
plotroc(testT, testY);