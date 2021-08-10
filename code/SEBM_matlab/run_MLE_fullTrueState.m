% MLE with full true state.  Compute the svd of Fisher matrix
%To test singularity of the Fisher information matrix >>> ill-posedness
% Last updated: Fei Lu, 2019-1-21

close all;  clear all  
restoredefaultpath;   addpaths; 


runMLE = 1; 
%% basic setttings 
settings_ALL;    % set prior, sampler, observation, and FEM   
rng(12);

%% generate data
plotON = 1 ;  saveON = 0; thetaTrue = sampleThetaTrue(prior); 
% obsPar.tN    = 100;
% obsPar.nodes = 1:obsPar.stateDim; 
[obs,Utrue] = generateDataNodes(thetaTrue,obsPar,femPar,'datafilename',saveON,plotON); 
figure; plot(Utrue(3,:)); temp = [mean(Utrue(3,:)), std(Utrue(3,:))];
fprintf('[Mean, Std] of priorMean u(3,t): %2.4f %2.4f\n',temp) ; 

%% compute the MLE and svd 
K    = length(thetaTrue); 
[mle,svdc1inv,stdFest] = MLE_truestate(Utrue,femPar,K); 

type =  '%2.4f   %2.4f  %2.4f'; 
fprintf(['MLE with tN=%5i      ',  type, '\n'], obsPar.tN, mle'); 
fprintf(['True theta            ', type, '\n'], thetaTrue); 
fprintf('stdF  %2.4f, and its estimator: %2.4f \n\n',femPar.stdF, stdFest); 

type =  '%2.4f   %2.4g  %2.4g';
fprintf(['svd (Fisher matrix) = ', type, '\n'], svdc1inv);  

ksdensity(Utrue(3,:));

%% results:    2019-1-22, with forward_solver_sphere2.m, 
%    - Gn exactly matching inference; no Fisher svd regularization
%% Gaussian prior,rng(12) (only matters with the true theta)
%{
MLE with tN=  100      9.9399     1.9555 -11.6315
True theta            29.2851   -23.5516  -5.4669
stdF  0.0325, and its estimator: 0.0312 
svd (Fisher matrix) = 5190.4653   0.02478  3.007e-06

MLE with tN= 1000     25.5563   -18.8739  -6.4215
True theta            29.2851   -23.5516  -5.4669
stdF  0.0325, and its estimator: 0.0325 
svd (Fisher matrix) = 5236.2648   0.007613  4.355e-07

MLE with tN=10000     23.8207   -16.4365  -7.1203
True theta            29.2851   -23.5516  -5.4669
stdF  0.0325, and its estimator: 0.0325 
svd (Fisher matrix) = 5241.2034   0.005649  1.384e-07

MLE with tN=100000    34.2212   -30.1107  -3.8442
True theta            29.2851   -23.5516  -5.4669
stdF  0.0325, and its estimator: 0.0325 
svd (Fisher matrix) = 5241.6688   0.005456  1.041e-07
%}
%% Unif prior, rng(12) (only matters with the true theta)
%{
MLE with tN=  100    -16.8848    37.0837  -20.8781
True theta            28.4035   -23.4184  -5.6822
stdF  0.0325, and its estimator: 0.0312 
svd (Fisher matrix) = 4851.2999   0.1452  3.295e-06

MLE with tN= 1000       6.0801   6.4122   -13.1819
True theta            28.4035   -23.4184  -5.6822
stdF  0.0325, and its estimator: 0.0325 
svd (Fisher matrix) = 4885.2895   0.02011  9.338e-07

MLE with tN=10000      18.4969   -10.1413  -9.0510
True theta             28.4035   -23.4184  -5.6822
stdF  0.0325, and its estimator: 0.0325 
svd (Fisher matrix) = 4889.0038   0.006644  3.058e-07

MLE with tN=100000      31.2987   -27.3513  -4.6428
True theta              28.4035   -23.4184  -5.6822
stdF  0.0325, and its estimator: 0.0325 
svd (Fisher matrix) = 4889.3497   0.005296  1.328e-07
%}

%% results:    2019-1-18, with foward--1
%{
MLE with tN=  100      4.3903    2.0382   -6.2330
True theta            30.0374   -23.9608  -5.6729
stdF  0.03251, and its estimator: 0.03222 
svd (Fisher matrix) = 5245.6928   0.09013  2.379e-06

MLE with tN= 1000      4.2389    1.8344   -5.8846
True theta            30.0374   -23.9608  -5.6729
stdF  0.03251, and its estimator: 0.03261 
svd (Fisher matrix) = 5286.0219   0.01867  6.806e-07


MLE with tN=10000      4.1074     1.7268  -5.6518
True theta            30.0374   -23.9608  -5.6729
stdF  0.03251, and its estimator: 0.03245 
svd (Fisher matrix) = 5290.5674   0.01089  4.226e-07

MLE with tN=100000      4.1254    1.7312  -5.6734
True theta            30.0374   -23.9608  -5.6729
stdF  0.03251, and its estimator: 0.03250 
svd (Fisher matrix) = 5290.9583   0.01002  3.717e-07
%}
