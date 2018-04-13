clear all
clc

load data3.mat;

%fit with least square
Xls =[ones([50,1]),X];
beta=inv(Xls'*Xls)*Xls'*Y

errM = Xls*beta-Y;
error = errM'*errM;
disp(['Error of LS: ', num2str(error)]);

%mostrar grafico de LS
x_prt1 = 1:50;
vYLS = beta(1)*Xls(:,1) + beta(2)*Xls(:,2) + beta(3)*Xls(:,3) +beta(4)*Xls(:,4);
figure;
plot(x_prt1, vYLS,'b');
hold on;
plot(x_prt1, Y, 'xr');
title('LS fit');

%fit with lasso
[B,FitInfo] = lasso(X,Y);
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log');
hold on; scatter(0.1,2.9388,'b', 'filled')
scatter(0.1,0.0648,'r','filled')
scatter(0.1,1.4252,'y','filled')
title('Lasso plot for Lasso fit');
lambda = 0.1;
[B, LassoInfor] = lasso(X,Y,'Lambda',lambda);
LassoInfor.Intercept
B

errM = X*B-Y+mean(Y);
error = errM'*errM;
disp(['Error of lasso: ', num2str(error)]);

%mostrar grafico de lasso
vYLASSO = LassoInfor.Intercept + B(1)*X(:,1) + B(3)*X(:,3);
figure;
plot(x_prt1, vYLASSO,'b');
hold on;
plot(x_prt1, vYLS,'r');
plot(x_prt1, Y, 'xr');
title('Lasso fit');

%fit with ridge
lambdavec = linspace(0.0001,10,100);
b = ridge(Y,X,lambdavec);
figure;
semilogx(lambdavec, b(1,:), 'b');
hold on;
semilogx(lambdavec, b(2,:), 'r');
semilogx(lambdavec, b(3,:), 'y');
hold on; scatter(0.1,2.9388,'b', 'filled')
scatter(0.1,0.0648,'r','filled')
scatter(0.1,1.4252,'y','filled')
title('Ridge coef. as function of lambda');

b = ridge(Y,X,0.1,0)

X1 =[ones([50,1]),X];
errM = X1*b-Y;
error = errM'*errM;
disp(['Error of ridge: ', num2str(error)]);

%mostrar grafico de ridge
vYRI = b(1) + b(2)*X(:,1) + b(3)*X(:,2) + b(4)*X(:,3);
figure;
plot(x_prt1, vYRI,'b');
hold on;
plot(x_prt1, vYLS,'r');
plot(x_prt1, Y, 'xr');
title('Ridge fit');

