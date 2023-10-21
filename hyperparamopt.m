%% Hyperparameter Optimization
% William Baumchen
close all; clear; clc

% Verbose Output - [0 for suppression, 1 for iteration]
verboze = 1;

%% Data Preprocessing

% Import Data
datain = readtable("winequality-white.csv");
% Shuffle Data Entries for Splitting Data
% Set random seed for reproducibility
rng(12)
datain = datain(randperm(size(datain,1)),:);
% Set Fraction of Entries for Test Set
a = 0.2;
% Split Data
Test = datain(1:round(a*size(datain,1)),:);
Train = datain(round(a*size(datain,1))+1:end,:);

%% Bayesian Optimization for Regression Models
% Set max evaluation 
bayesianOptions = struct('MaxObjectiveEvaluations',300,'Verbose',verboze);
% Fit set of regression models using bayesian optimization
[bayesianMdlr,bayesianResults] = fitrauto(Train,"quality","HyperparameterOptimizationOptions",bayesianOptions);
% Plot confusion chart for Test values
figure(1)
confusionchart(Test.quality,round(predict(bayesianMdlr,Test)))
title("Bayesian Optimization - Classification")
% Find MSE of Test values 
bayesianAccuracyr = loss(bayesianMdlr,Test,"quality");
disp(['Current Accuracy (MATLAB MSE): ',num2str(bayesianAccuracyr)]);

%% Bayesian Optimization for Classification Models
% Set max evaluation 
bayesianOptions = struct('MaxObjectiveEvaluations',300,'Verbose',verboze);
% Fit set of classification models using bayesian optimization
[bayesianMdlc,bayesianResults] = fitcauto(Train,"quality","HyperparameterOptimizationOptions",bayesianOptions);
% Plot confusion chart for Test values
figure(2)
confusionchart(Test.quality,round(predict(bayesianMdlc,Test)))
title("Bayesian Optimization - Regression")
% Find MSE of Test values 
bayesianAccuracyc = loss(bayesianMdlc,Test,"quality");
disp(['Current Accuracy (MATLAB MSE): ',num2str(bayesianAccuracyc)]);