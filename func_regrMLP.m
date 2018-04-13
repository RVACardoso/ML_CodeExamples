clear all;
clc;

load regression_data.mat;
net = fitnet(40);
net.layers{2}.transferFcn='purelin';
%{
net.divideFcn='divideind';
net.divideParam.trainInd=1:70;
net.divideParam.valInd=71:85;
net.divideParam.testInd=86:100;
%}
net.performFcn='mse'; 
net.trainParam.epochs=10000; % max epochs
net.trainParam.goal=0.005; % training goal 
%{
[net,tr] = train(net,X,T);

scatter(X(1:70),T(1:70),'red')
hold on
scatter(X(86:100),T(86:100),'blue')
xx = linspace(-1,1);
yy = net(xx);
plot(xx,yy)
%}
net.divideFcn='divideind';
net.divideParam.trainInd=1:85;
net.divideParam.testInd=86:100;

[net,tr] = train(net,X,T);
%yy1 = net(xx);
hold on;
%plot(xx,yy1)

