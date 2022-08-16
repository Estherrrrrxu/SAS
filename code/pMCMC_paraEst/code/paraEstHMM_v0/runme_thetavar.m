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

clear all; clc; 
% addpath([pwd, '/results/MLE_tN1k/']) ; 

%%%===== Using fixed rng in line 38 
ssmPar.prior = 2;      % prior of para: 1 for Guassian, 2 for uniform
ssmPar.tN  = 100;      % number of time steps in state space model
model_settings();      % load the model settings: true para from prior

%%  Select the parameters to be estimated
ssmPar.thetaInd = 1:2;    % indices of theta to be estimated
ssmPar.thetaV = 'thetaV12'; % estimate theta, V or thetaV.  

% MLE_trueX(ssmPar);     % MLE estimator from true states --- identiability

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
if exist(sampleFilename,'file') == 0  % if no sample file, generate samples
     saveON = 1;     plotOn =1; 
    %% generate Observation data
    if exist(Obsdatafile,'file') == 0
        [Xtrue,obs] = generate_data(ssmPar,ssmPar.Phi_n,plotOn);  
        if saveON ==1;  save(Obsdatafile);  end
    end
    load(Obsdatafile,'obs','Xtrue');  % Data file includes ssmPar sampleFilename etc
    
    %%  %%%%%  Sample the posterior by particle MCMC: PGAS
    % Set up parameter of PGAS
    Np      = 20;             % Number of particles used in PGAS
    numMCMC = 10000;           % Number of iterations in the MCMC samplers
    burnin  = 3000;            % Number of interations to burn
    plotOn  = 1;              % Plot intermediate sample paths?
    % Run the pgas algorithm:   size(X)=[Dx,tN,numMCMC]; theta: [K,numMCMC]
    fprintf('Running PGAS (N=%i). Progress: ',Np); tic;
    [Xsample,thetaSample,varV,ess] = pgas_Bayes(numMCMC,obs,Np,ssmPar);   
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




function MLE_trueX(ssmPar)
% Estimate the parameters by MLE from true X and V
K      = 1;  
mLag   = max(ssmPar.p, ssmPar.q); 
sigmaV = ssmPar.sigmaV; 

ssmPar.tN  = 1000;     % number of time steps in state space model
X0 = randn(K,mLag);   V0 = randn(size(X0))*sigmaV;
[Xtrue, Vtrue] = StateModel(X0,V0,ssmPar,ssmPar.Phi_n);
%  plot(Xtrue); 

% Xtrue =  Xtrue  + 0.2*randn(size(Xtrue)); 

tN    = ssmPar.tN;  Terms = ssmPar.Terms; fnt   = ssmPar.fnt; 
tms = zeros(length(ssmPar.thetaTrue),tN-mLag); 
Xft = zeros(1,tN-mLag); 
for t = mLag+1:tN             % estimate the terms in likelihood
    XX = Xtrue(:,t-mLag+1:t); 
    VV = Vtrue(:,t-mLag+1:t);
    tms(:,t) = Terms(XX,VV);
    Xft(:,t) = Xtrue(t) - fnt(t);  
end

%% the MLE: Ax = b  
A      = tms * tms';   b  = tms * Xft'; % theta = A\b;       
estInd = ssmPar.thetaInd; knownInd = setdiff(1:4,estInd); 
b     =  b(estInd)- A(estInd,knownInd)*ssmPar.thetaTrue(knownInd);
A     = A(estInd,estInd); 
theta = A\b;          %  MLE 
sV    = var(Vtrue); 

disp('True (top row) and MLE coefficients:');
disp([ssmPar.thetaTrue(estInd)'; theta']);  
fprintf('True and MLE sigmaV: %4.2f %4.2f \n', ssmPar.sigmaV, sV); 
end 




