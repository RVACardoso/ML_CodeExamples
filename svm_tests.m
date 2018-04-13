clear all
clc

load spiral.mat;

result1 ={'p', 'SV_number', 'error%'};
for p=1:25
    SVMStruct = svmtrain(X,Y,'kernel_function', 'polynomial','polyorder',p,'method', 'QP','boxconstraint',10^4, 'showplot', true); 
    SVnumber = length(SVMStruct.SupportVectors);
    class = svmclassify(SVMStruct,X);
    erro=(sum(Y~=class)/length(Y))*100;
    temp= [p  SVnumber  erro];
    result1 =[result1; num2cell(temp)];
end
result1

result2 ={'sigma', 'SV_number', 'error%'};
for sigma=0.1:0.1:2
    SVMStruct = svmtrain(X,Y,'kernel_function', 'rbf','rbf_sigma',sigma,'method', 'QP','boxconstraint',10^4, 'showplot', true); 
    SVnumber = length(SVMStruct.SupportVectors);
    class = svmclassify(SVMStruct,X);
    erro=(sum(Y~=class)/length(Y))*100;
    temp= [sigma  SVnumber  erro];
    result2 =[result2; num2cell(temp)];
end
result2
  
load chess33.mat;
result3 ={'sigma', 'SV_number', 'error%'};
for sigma=0.1:0.1:6
    SVMStruct = svmtrain(X,Y,'kernel_function', 'rbf','rbf_sigma',sigma,'method', 'QP','boxconstraint',Inf, 'showplot', false); 
    SVnumber = length(SVMStruct.SupportVectors);
    class = svmclassify(SVMStruct,X);
    erro=(sum(Y~=class)/length(Y))*100;
    temp= [sigma  SVnumber  erro];
    result3 =[result3; num2cell(temp)];
end
result3

sigma=1.0;
load chess33.mat;
SVMStruct = svmtrain(X,Y,'kernel_function', 'rbf','rbf_sigma',sigma,'method', 'QP','boxconstraint',Inf, 'showplot', true); 
  xlabel('x1');
    ylabel('x2');
SVnumber = length(SVMStruct.SupportVectors)
class = svmclassify(SVMStruct,X);
erro=(sum(Y~=class)/length(Y))*100

load chess33n.mat;
sigma=1.0;
figure;
SVMStruct = svmtrain(X,Y,'kernel_function', 'rbf','rbf_sigma',sigma,'method', 'QP','boxconstraint',Inf, 'showplot', true); 
  xlabel('x1');
    ylabel('x2');
SVnumber = length(SVMStruct.SupportVectors)
class = svmclassify(SVMStruct,X);
erro=(sum(Y~=class)/length(Y))*100

load chess33n.mat;
sigma=1.0;
result4 ={'exponent', 'SV_number', 'error%'};
for g=-4:15
    SVMStruct = svmtrain(X,Y,'kernel_function', 'rbf','rbf_sigma',sigma,'method', 'QP','boxconstraint',10^g, 'showplot', true); 
  
    SVnumber = length(SVMStruct.SupportVectors);
    class = svmclassify(SVMStruct,X);
    erro=(sum(Y~=class)/length(Y))*100;
    temp= [g  SVnumber  erro];
    result4 =[result4; num2cell(temp)]
end
result4;
