% Compute statistics of estimators:  (Observe partial nodes!!!)
%  - generate datasets with thetatrue from prior; 
%  - get estimator for each dataset; 
%  - compute the stats of estimators
% Last updated: Fei Lu, 2019/3/20
%{
  Time: 23 hrs for  tN=100,numMCMC=10000
%}

close all;     addpaths; 
   
stateRegul =1;  % regulate the posterior of the states by 
if stateRegul==1; SR = 'SR_crps'; else; SR = ''; end 
%% basic setttings 
settings_ALL_1k;    % set prior, sampler, observation, and FEM   
     
%% output file names
if prior.flag ==0;         priorName = 'GaussPrior/';   % 0 Gaussian, 2 uniform
elseif prior.flag == 2;    priorName = 'UnifPrior/';     end    
nodeNum = sprintf('nodes%i_',length(obsPar.nodes)); nodeNum = [nodeNum,priorName]; 
% if obsPar.stateDim ==42;  
if strcmp( pmcmcPar.sampler, 'IPF')
      datapath = [home_path, 'output/nlfnDeg014_IPF/',nodeNum]; 
else; datapath = [home_path, 'output/nlfnDeg014_SIR/',nodeNum]; 
end
addpath(datapath);

if pmcmcPar.numMCMC == 1000; type = 'MC1k'; else;  type = 'MC10k'; end
if pmcmcPar.pgas ==1;      type1 = 'cpfas.mat';  % with Ancestor sampling      
elseif pmcmcPar.pgas ==0;  type1 = 'cpf.mat';  
end

Datafilename  = [datapath,SR,'stats_tN100',type,type1];   
 
%% Generate observation data and Sample the posterior by particle MCMC 
numEns = 100; K = length(prior.mu);
errThetaMeanPost  = zeros(numEns, K); 
errThetaMaxPost   = errThetaMeanPost; 
errRelState  = zeros(numEns,1); 
errTrajMean  = zeros(numEns, tN+1 );  
meanPost     = zeros(3,numEns);  MAP = meanPost; 
covPost      = zeros(3,3,numEns);

% coverage frequency, CRPS, energy_score
    quantile = 0.9; 
cf_theta      = zeros(1,numEns);     cf_state      = zeros(1,numEns);  
crps_theta    = zeros(3,numEns);     crps_state    = zeros(1,numEns);
es_theta      = zeros(1,numEns);     es_state      = zeros(1,numEns);
    
    
settings = sprintf(', tN = %i, # Simulations= %i',tN, numEns);  
fprintf(['Running: Prior= ', type, ', terms= ', nldeg, settings, '\n \n']); 
ss = sprintf('Time estimate: 2.5min for single simulation with tN=100,numMCMC=1000'); 
fprintf([ss,'\n              ==>> About 2.5*100/4 = 62.5 min \n']);  %% tN1k+MC1k: Time=15.35 hour
disp(datetime('now')); 

    %  %%%%%  Sample the posterior by particle MCMC: PGAS
    Np      = pmcmcPar.Np;         % number of particles in SMC
    numMCMC = pmcmcPar.numMCMC;  % length of MCMC chain
    burnin  = .3*numMCMC;
tic 

%% parfor loop! 
parfor j = 1:numEns
    plotON =0;      saveON = 0;        
    thetaTrue   = sampleThetaTrue(prior); 
    [obs,Utrue] = generateDataNodes(thetaTrue,obsPar,femPar,'null',saveON,plotON);  
  
    % Run the algorithms:   size(Usample)=[ Dx,tN,numMCMC]; theta: [K,numMCMC]
    progressON = 0; 
    if stateRegul
         [Usample,ess,theta] = pgas_stateParNodes_statePrior(pmcmcPar,obs,obsPar,femPar,prior,progressON);
    else
         [Usample,ess,theta] = pgas_stateParNodes(pmcmcPar,obs,obsPar,femPar,prior,progressON);
    end      
    
    %% stats of parameter: MAP and posterior mean, coverageFreq, CRPS
    % size(theta)=K x numMCMC
    theta = theta(:,burnin:numMCMC);
    meanPosterior = mean(theta,2);      % a column vector
    covPosteior   = cov(theta')';    
    maxpost       = findMAP(theta);
    
    meanPost(:,j) = meanPosterior; MAP(:,j) = maxpost;  
    covPost(:,:,j)= covPosteior; 
    errThetaMeanPost(j, :) = meanPosterior' - thetaTrue; 
    errThetaMaxPost(j,:)   = maxpost' - thetaTrue; 
    
    [~,es_theta(j),crps_theta(:,j),~]  = probab_scores(theta,thetaTrue',[],[]);
    % calculate coverage frequency of theta: not so useful. Good for highD state (maybe)
    cf_theta(j) = coverageFrequency(theta, quantile,thetaTrue);

    %% stats of state: relative error in state estimation by  posterior mean
    samples = Usample(:,:,burnin:numMCMC); 
    sMean    = mean(samples, 3);
    % sStd     = std(samples,0, 3);
    relerr_temp = abs( (Utrue-sMean)./Utrue);   % KxtN
    temp        = mean(relerr_temp,1);
    errTrajMean(j,:) = temp; 
    errRelState(j,:) = mean(temp);

    [crps_state(j),es_state(j),~,~]  = probab_scores(samples,Utrue,[],[]);
    
    samples = reshape( samples,[],length(samples(1,1,:)) ) ;
    clmf    = reshape(Utrue,[],1); 
    cf_state(j) = coverageFrequency(samples, quantile,clmf);
    
    
    if mod(j, numEns/10) ==1
        fprintf('Progress: %3i/%3i ', j,numEns);     disp(datetime('now')); 
    end
end
timeElapsed = ceil(toc); fprintf('Time elapsed %2.2f hours\n', toc/3600)
save(Datafilename, 'errThetaMeanPost','errRelState','errTrajMean','prior', 'MAP', ...
                  'meanPost','covPost','pmcmcPar','errThetaMaxPost','timeElapsed',...
                  'crps_state','es_state','cf_state','crps_theta','es_theta','cf_theta'); 



function MAP = findMAP(sampleArray)
% find the MAP of the posterior: in each marginal distribution
[K,~] = size(sampleArray); 
MAP = zeros(K,1);
for kk=1:K
  samples = sampleArray(kk,:); 
  minS    = min(samples); maxS  = max(samples); gap =(maxS-minS)/400; 
  points  = minS-gap:gap:maxS+gap; 
  [f,xi]  = ksdensity(samples, points);  
  [~,indMax] = max(f); MAP(kk)=xi(indMax); 
end
end




