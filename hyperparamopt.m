%% Hyperparameter Optimization
% William Baumchen
close all; clear; clc
% Verbose Output - [0 for suppression, 1 for iteration]
verboze = 1;
% Show Plots - [0 for suppression, 1 for iteration]
plotz = 1;
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
Test = datain(1:round(a*size(datain,1)),:);
Train = datain(round(a*size(datain,1))+1:end,:);

%% Bayesian Optimization for Regression Models
% Set max evaluation 
bayesianOptions = struct('MaxObjectiveEvaluations',iternn,'Verbose',verboze,'Repartition',1,'ShowPlots',plotz);
% Fit set of regression models using bayesian optimization
[bayesianMdlr,bayesianResultsr] = fitrauto(Train,"quality","HyperparameterOptimizationOptions",bayesianOptions);

%% Bayesian Optimization for Classification Models
% Set max evaluation 
bayesianOptions = struct('MaxObjectiveEvaluations',iternn,'Verbose',verboze,'Repartition',1,'ShowPlots',plotz);
% Fit set of classification models using bayesian optimization
[bayesianMdlc,bayesianResultsc] = fitcauto(Train,"quality","HyperparameterOptimizationOptions",bayesianOptions);

%% Test Against Baseline

% Fit a new set of baseline models
baseR = fitrensemble(Train,"quality");
baseC = fitcecoc(Train,"quality");
basecacc = loss(baseC,Test,"quality");
baseracc = loss(baseR,Test,"quality");
% Get current loss metrics on models:
disp(['Baseline Classification Loss: ',num2str(basecacc)]);
disp(['Baseline Regression MSE: ',num2str(baseracc)]);
bayesianAccuracyc = loss(bayesianMdlc,Test,"quality");
disp(['Current Model Classification Loss: ',num2str(bayesianAccuracyc)]);
bayesianAccuracyr = loss(bayesianMdlr,Test,"quality");
disp(['Current Model Regression MSE: ',num2str(bayesianAccuracyr)]);

% Plot confusion chart for Test values
figure(1);
confusionchart(Test.quality,round(predict(bayesianMdlr,Test)))
title("Bayesian Optimization - Classification")
% Plot confusion chart for outer Test values
figure(2);
confusionchart(Test.quality,round(predict(bayesianMdlc,Test)))
title("Bayesian Optimization - Regression")

%% Save Resulting Data
save('hpamtopt.mat')