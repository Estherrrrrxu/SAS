%  Estimate parameter by Particle Gibbs with Ancestor Sampling (PGAS) algorithm,
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
%  Fei Lu Last updated 2018-6-23     feilu@math.jhu.edu
 

%%%===== Using fixed rng in line 38 
ssmPar.thetaV = 'theta'; % estimate theta only. 

tN     = 1000;     % number of time steps in state space model
type   = strcat('evenObs_unif_tN',num2str(tN)); ssmPar.prior = 2;     % uniform prior
% type   = 'Odd_GaussNp20'; ssmPar.prior  = 1;  % Gaussian prior

sampletype= strcat(type,'_Np20nMC10k'); 
Obsdatafile    = strcat('Obsdata_',type,'.mat'); % e.g. 'Obsdata_tN100.mat';
figname        = sampletype;      
sampleFilename = strcat('sampledata_',sampletype,'.mat'); % e.g 'sampledata_tN100.mat';  


%%  Generate observation data and Sample the posterior by particle MCMC: PGAS
% -- the true parameter is sampled from a prior: good for Bayesian tests
% -- Can also be used in maximum likelihood tests
if exist(sampleFilename,'file') == 0  % if no sample file, generate samples
     saveON = 1; rng(12); 
    if ssmPar.prior ==1    %%% === Gaussian prior of parameters
        mu  = [0.9;-0.2;25;0.8]; % [a1; a2; b1; d1]  % A COLUMN vector
              % a1,a2 linear stability: roots of 1-a1*z - a2*z^2 outside of unit ball.
        std = [0.2; 0.1; 3; 0.2];  % sigma_i
        thetaTrue = mu + 0*randn(size(mu)).*std; 
%        thetaTrue = [0.9,-0.2,25,0.8];  
        ssmPar.priormu   = mu;           ssmPar.priorstd = std; 
        ssmPar.thetaTrue = thetaTrue; 
    elseif ssmPar.prior ==2     %%% === Uniform prior
        lb  = [0.7; -0.4;20; -0.2];    ssmPar.priorlb = lb; 
        ub  = [0.96; -0.1; 30; 0.9];   ssmPar.priorub = ub;
        thetaTrue = rand(size(lb)).*(ub-lb) + lb;  
        ssmPar.thetaTrue  = thetaTrue;  % Here prior.mu used to pass the true value.
    end
    ssmPar.priorsVa = 0.1;   ssmPar.priorsVb = 0.1;  % Inverse Gamma for sV
    %% generate Observation data
    if exist(Obsdatafile,'file') == 0
        model_data(tN,ssmPar,Obsdatafile, saveON); 
    end
    load(Obsdatafile); % Data file includes info of the state space model 
    
    %%  %%%%%  Sample the posterior by particle MCMC: PGAS
    % Set up parameter of PGAS
    Np = 20;                  % Number of particles used in PGAS
    numMCMC = 10000;          % Number of iterations in the MCMC samplers
    burnin  = 3000;            % Number of interations to burn
    plotOn  = 1;             % Plot intermediate sample paths?
    % Run the pgas algorithm:   size(X)=[Dx,tN,numMCMC]; theta: [K,numMCMC]
    fprintf('Running PGAS (N=%i). Progress: ',Np); tic;
    [Xsample,thetaSample,varV,ess] = pgas_theta(numMCMC,obs,Np,ssmPar);   % size(theta)=K x numMCMC 
    timeelapsed = toc;
    fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed); 
    clearvars -except timeelapsed Xsample thetaSample varV ess numMCMC burnin ...
                      thetaTrue Xtrue  Np sampleFilename figname ssmPar;     
    save(sampleFilename);
end

%% represent the results
% plots: marginal of theta, ensemble trajectory, update rate
printFig = 1; 
Plots_marginal(printFig, sampleFilename,figname);







