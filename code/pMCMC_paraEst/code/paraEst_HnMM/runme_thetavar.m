%  Estimate the VARIANCEs by Particle Gibbs with Ancestor Sampling (PGAS) algorithm,
%  References:
%   [1] F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for
%   Particle Gibbs", Proceedings of the 2012 Conference on Neural
%   Information Processing Systems (NIPS), Lake Taho, USA, 2012.
% and
%   [2] C. Andrieu, A. Doucet and R. Holenstein, "Particle Markov chain Monte
%   Carlo methods" Journal of the Royal Statistical Society: Series B,
%   2010, 72, 269-342.
%
% The script generates a batch of data y_{1:T} from a nonlinear time series model,
%
%   x_{t+1} = a_1x_t + a_2x_{t-1}+ b1 x_t/(1+x_t^2)+8\cos(1.2t)+d_1v_{t-1}+ v_t,
%   y_t = 0.05*x_t^2 + w_t,
%
% with v_t ~ N(0,sV) and w_t ~ N(0,sW). 
% We assume that sV, sW are given, but the parameters 
%           theta = [a1,a2, b1,d1]
% are unknown and are to be estimated.  
% The PGAS algorithm generates a Markov chain to sample the posterior  
% of the parameter and state: p(theta, x_{1:T} | y_{1:T}). 
%
%  Fei Lu Last updated 2018-6-27     feilu@math.jhu.edu

clear all; close all; 
addpaths; 

%%%===== Using fixed rng in line 38 
ssmPar.prior = 1;      % prior of para: 1 for Guassian, 2 for uniform
ssmPar.tN  = 10000;      % number of time steps in state space model
model_settings();      % load the model settings: true para from prior

%%  Select the parameters to be estimated
ssmPar.thetaInd = [1];     % indices of theta to be estimated  [1,2,3,4] -- a1,a2,b1,d1
ssmPar.thetaV = 'thetaV1'; % estimate theta, V or thetaV.  


truestate = 0;
if truestate
    MLE_trueX(ssmPar);     % MLE estimator from true states --- identiability
    % return
end

%% filenames for data/figure 
if     ssmPar.prior == 2;  type = 'evenObs_unif';      % uniform prior
elseif ssmPar.prior == 1;  type = 'evenObs_Gauss';      % Gaussian prior
end
sampletype     = strcat('_bayes_Np20MC10k_',ssmPar.thetaV,type); 
Obsdatafile    = strcat('Obsdata',sampletype,'.mat'); % e.g. 'Obsdata_.mat';
figname        = sampletype;      
sampleFilename = strcat('samples',sampletype,'.mat'); % e.g 'samples_.mat';  

%%  Generate observation data and Sample the posterior by particle MCMC: PGAS
% -- the true parameter is sampled from a prior: good for Bayesian tests
% -- Can also be used in maximum likelihood tests
%% generate Observation data
 saveON = 1;     plotOn =1; 
if exist(Obsdatafile,'file') == 0
    [Xtrue,obs] = generate_data(ssmPar,ssmPar.Phi_n,plotOn);  
    if saveON ==1;  save(Obsdatafile);  end
end
load(Obsdatafile,'obs','Xtrue');  % Data file includes ssmPar sampleFilename etc

sampleFilename2 = strcat('OnlineEM',sampletype,'.mat');

%% %% sample by online EM using SIR-FS
if exist(sampleFilename2,'file') == 0  % if no sample file, generate samples
    Np      = 100;             % Number of particles  
    fprintf('Running OnlineEM with SIR-FS (N=%i). Progress: ',Np);    tic;
   [Xsample,vp,w,thetaSample,varV,ess] = SIR_FS_onlineEM(obs,ssmPar,Np);    
   
    timeelapsed = toc;        
    fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed); 
    clearvars -except timeelapsed Xsample thetaSample varV ess vp w ...
                      thetaTrue Xtrue  Np sampleFilename2 figname ssmPar;     
    save(sampleFilename2);
end


%%  %%%%%  Sample the posterior by particle MCMC: PGAS
 if exist(sampleFilename,'file') == 0  % if no sample file, generate samples
    % Set up parameter of PGAS
    Np      = 100;             % Number of particles used in PGAS
    numMCMC = 1000;           % Number of iterations in the MCMC samplers
    burnin  = 300;            % Number of interations to burn
    plotOn  = 1;              % Plot intermediate sample paths?
    % Run the pgas algorithm:   size(X)=[Dx,tN,numMCMC]; theta: [K,numMCMC]
    fprintf('Running PGAS (N=%i). Progress: ',Np); tic;
%    [Xsample,thetaSample,varV,ess] = pgas_Bayes(numMCMC,obs,Np,ssmPar);  
[Xsample,thetaSample,varV,ess] = pgas_MLE_refpath(numMCMC,obs,Np,ssmPar);
%   [Xsample,thetaSample,varV,ess] = pgas_SAEM(numMCMC,obs,Np,ssmPar);
    timeelapsed = toc;        
    fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed); 
    clearvars -except timeelapsed Xsample thetaSample varV ess numMCMC burnin ...
                      thetaTrue Xtrue  Np sampleFilename figname ssmPar;     
    save(sampleFilename);
 end

%% represent the results
% plots: marginal of theta, ensemble trajectory, update rate
printFig = 1; 
Plots_marginal(); % (printFig, sampleFilename,figname);






