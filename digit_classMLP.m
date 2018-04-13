clear all;
clc;

load digits;
net = patternnet([15]); 
net.performFcn='mse'; 
net.layers{1}.transferFcn='tansig';
net.layers{2}.transferFcn='tansig';
net.divideFcn='divideind';
net.divideParam.trainInd=1:400;
net.divideParam.testInd=401:560;

net.trainFcn = 'traingdx'  %traingdm  traingdx
net.trainParam.lr=4.5; % learning rate    4.5  0.5
net.trainParam.mc=0.7; % Momentum constant
net.trainParam.show=10000; % # of epochs in display
net.trainParam.epochs=10000; % max epochs
net.trainParam.goal=0.05; % training goal 

tic
[net,tr] = train(net,X,T);
toc

x_test=X(:,tr.testInd);
t_test=T(:,tr.testInd);
y_test = net(x_test);
plotconfusion(t_test,y_test);
   
