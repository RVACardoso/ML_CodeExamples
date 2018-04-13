clear all
clc

load data2.mat;
xi=x;
yi=y;

[len_x, a] = size(xi);
[len_y, a] = size(yi);

if len_x==len_y
    n=len_x;
else
    disp('Número de inputs é diferente do número de outputs');
end

x=[ones([n,1])];

p = input('Insira o grau do polinómio(P)');
if n<=p
    disp('Número de pontos tem de ser maior que P');
    return;
end

for j = 1:p
    x_temp = xi.^j;
    x=[x,x_temp];
end

beta=inv(x'*x)*x'*yi

%calcular erro
errM = x*beta-y;
error = errM'*errM;
disp(['Error: ', num2str(error)]);

%show graph
x_prt = linspace(min(xi), max(xi), 100);

x_pot = [ones([100,1]) x_prt'];
for j = 2:p
   x_pot = [x_pot, x_pot(:,end).*x_prt']; 
end
y_prt = x_pot*beta;

plot(xi, yi, 'xr', x_prt, y_prt, 'b')







