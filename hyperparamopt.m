%% Hyperparameter Optimization
% William Baumchen
close all; clear; clc
% Verbose Output - [0 for suppression, 1 for iteration]
verboze = 0;
% Show Plots - [0 for suppression, 1 for iteration]
plotz = 0;
% Iteration Budget
iternn = 300;

%% Data Preprocessing

% Import Data
datain = readtable("winequality-white.csv");
% Shuffle Data Entries for Splitting Data
% Set random seed for reproducibility
rng(42)
datain = datain(randperm(size(datain,1)),:);
% Set Fraction of Entries for Test Set
a = 0.2;
% Split Data
Test1 = datain(1:round(a*size(datain,1)),:);
Train1 = datain(round(a*size(datain,1))+1:end,:);
% Set Fraction of Training Entries for Training Test Set
a = 0.15;
% Split Data
Test = Train1(1:round(a*size(Train1,1)),:);
Train = Train1(round(a*size(Train1,1))+1:end,:);


%% Bayesian Optimization for Regression Models
% Set max evaluation 
bayesianOptions = struct('MaxObjectiveEvaluations',iternn,'Verbose',verboze,'Repartition',1,'ShowPlots',plotz);
% Fit set of regression models using bayesian optimization
[bayesianMdlr,bayesianResultsr] = fitrauto(Train,"quality","HyperparameterOptimizationOptions",bayesianOptions);
% Plot confusion chart for Test values
figure(1)
confusionchart(Test.quality,round(predict(bayesianMdlr,Test)))
title("Bayesian Optimization - Classification")
% Find MSE of Test values 
bayesianAccuracyr = loss(bayesianMdlr,Test,"quality");
disp(['Current Accuracy (MATLAB Loss): ',num2str(100*(1-(bayesianAccuracyr))),'%']);

%% Bayesian Optimization for Classification Models
% Set max evaluation 
bayesianOptions = struct('MaxObjectiveEvaluations',iternn,'Verbose',verboze,'Repartition',1,'ShowPlots',plotz);
% Fit set of classification models using bayesian optimization
[bayesianMdlc,bayesianResultsc] = fitcauto(Train,"quality","HyperparameterOptimizationOptions",bayesianOptions);
% Plot confusion chart for Test values
figure(2)
confusionchart(Test.quality,round(predict(bayesianMdlc,Test)))
title("Bayesian Optimization - Regression")
% Find MSE of Test values 
bayesianAccuracyc = loss(bayesianMdlc,Test,"quality");
disp(['Current Accuracy (MATLAB Loss): ',num2str(100*(1-(bayesianAccuracyc))),'%']);

%% Test Against Baseline

baseR = fitrensemble(Train1,"quality");
baseC = fitcensemble(Train1,"quality");
basecacc = loss(baseC,Test1,"quality");
baseracc = loss(baseR,Test1,"quality");
disp(['Baseline Classification Accuracy (MATLAB Loss): ',num2str(100*(1-(basecacc))),'%']);
disp(['Baseline Regression Accuracy (MATLAB Loss): ',num2str(100*(1-(baseracc))),'%']);

%% Save Resulting Data
% save('hpamtopt.mat')