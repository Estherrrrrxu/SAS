% stats of MLE with many simulations
% MLE with full true state.  Compute the svd of Fisher matrix
%To test singularity of the Fisher information matrix >>> ill-posedness
% Last updated: Fei Lu, 2019-1-21

close all;   
restoredefaultpath;   addpaths; 

runMLE = 1; 
%% basic setttings 
settings_ALL;    % set prior, sampler, observation, and FEM   
rng(12);
thetaTrue = sampleThetaTrue(prior); K    = length(thetaTrue); 

Nsim = 100;   tNseq = 10.^(2:5); L = length(tNseq);
mleTrue = zeros(K,Nsim,L); svdTrue = zeros(K,Nsim,L); stdF_True = zeros(Nsim,L);
mleObs  = zeros(K,Nsim,L); svdObs  = zeros(K,Nsim,L); stdF_Obs  = zeros(Nsim,L);
theta_true = zeros(K,Nsim);

fprintf('  MLE statistics with %3g simulations \n', Nsim); 
disp(datetime('now'));

obsPar.tN    = tNseq(end);  obsPar.nodes = 1:obsPar.stateDim;  % observe all nodes
plotON = 0 ;  saveON = 0;   
tic 
parfor n = 1:Nsim 
%% generate data
thetaTrue = sampleThetaTrue(prior); theta_true(:,n) = thetaTrue';
[obs,Utrue] = generateDataNodes(thetaTrue,obsPar,femPar,'datafilename',saveON,plotON);  

%% compute the MLE and svd
for ll=1:L
    utemp = Utrue(:,1:tNseq(ll));  
    [mle,svdc1inv,stdFest] = MLE_truestate(utemp,femPar,K); 
    mleTrue(:,n,ll) = mle; svdTrue(:,n,ll) = svdc1inv'; stdF_True(n,ll) = stdFest; 

    utemp = obs(:,1:tNseq(ll));  
    [mle,svdc1inv,stdFest] = MLE_truestate(utemp,femPar,K); 
    mleObs(:,n,ll) = mle; svdObs(:,n,ll) = svdc1inv'; stdF_Obs(n,ll) = stdFest; 
end
end
toc
save mle_stats_Unifprior.mat prior femPar mleTrue svdTrue stdF_True mleObs svdObs stdF_Obs...
                   theta_true   tNseq; 




