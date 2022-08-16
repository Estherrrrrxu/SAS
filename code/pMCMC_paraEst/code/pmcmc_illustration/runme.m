%%% Remark: if qinit = q0; rinit = r0=0.1, the pgas does not work due to
%%% degeneracy of weights in line93: w_as = w(:,t-1).*m = 0.  
% To make the PF to run, increment must not singular to the previous weight
% ---->>> what this implies on general condition? *****

% This script is an illustration of the Particle Gibbs with Ancestor
% Sampling (PGAS) and the Particle Marginal Metropolis-Hastings (PMMH)
% algorithms, presented in,
%
%   [1] F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for
%   Particle Gibbs", Proceedings of the 2012 Conference on Neural
%   Information Processing Systems (NIPS), Lake Taho, USA, 2012.
%
% and
%
%   [2] C. Andrieu, A. Doucet and R. Holenstein, "Particle Markov chain Monte
%   Carlo methods" Journal of the Royal Statistical Society: Series B,
%   2010, 72, 269-342.
%
% respectively.
%
% The script generates a batch of data y_{1:T} from the the standard
% nonlinear time series model,
%
%   x_{t+1} = 0.5*x_t + 25*x_t/(1+x_t^2) + 8*cos(1.2*t) + v_t,
%   y_t = 0.05*x_t^2 + e_t,
%
% with v_t ~ N(0,q) and e_t ~ N(0,r). The process noise and measurement
% noise variances (q,r) are treated as unkown parameters with inverse Gamma
% priors. The PGAS and PMMH algorithms are then executed independently to
% find the posterior parameter distribution p(q, r | y_{1:T}).
%
%
%   Fredrik Lindsten
%   Linköping, 2013-02-19
%   lindsten@isy.liu.se



%% PGAS and PMMH ==========================================================
% Set up some parameters
N1 = 5;                 % Number of particles used in PGAS
N2 = 5;               % Number of particles used in PMMH
T = 100;                % Length of data record
numMCMC = 300;          % Number of iterations in the MCMC samplers
burnin = 30;            % Number of interations to burn
plotOn  = 1;            % Plot intermediate sample paths?
close all; 
% Generate data
q0 = 0.1;  % True process noise variance
r0 = 0.1;    % True measurement noise variance
[x0,y0] = generate_data(T,q0,r0);

% Hyperparameters for the inverse gamma priors (uninformative)
prior.a = 0.01;
prior.b = 0.01;

% Parameter proposal for PMMH (Gaussian random walk)
prop.sigma_q = 1;
prop.sigma_r = 1;

% Initialization for the parameters
qinit = 1;      % if set to be q0, the PF blows up
rinit = 0.1;     

% Run the algorithms
fprintf('Running PGAS (N=%i). Progress: ',N1); tic;
[q_pgas, r_pgas, x_pgas] = pgas(numMCMC, y0, prior, N1, qinit, rinit, q0, r0, plotOn);
timeelapsed = toc;
fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed);

fprintf('Running PMMH (N=%i). Progress: ',N2); tic;
[q_pmmh, r_pmmh, x_pmmh] = pmmh(numMCMC, y0, prior, prop, N2, qinit, rinit, q0, r0, plotOn);
timeelapsed = toc;
fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed);tic;

%% Plot the results
figure(10);
subplot(2,1,1)
[fq,xq] = ksdensity(q_pgas(burnin+1:end));
plot(xq,fq,'b-'); hold on;
[fq,xq] = ksdensity(q_pmmh(burnin+1:end));
plot(xq,fq,'r-');
tmp=get(gca,'ylim');
plot([q0 q0],tmp*2,'k-');
set(gca,'ylim',tmp);
hold off;
legend('PGAS','PMMH');
title('Empirical PDF of p(q | y_{1:T})')

subplot(2,1,2)
[fr,xr] = ksdensity(r_pgas(burnin+1:end));
plot(xr,fr,'b-'); hold on;
[fr,xr] = ksdensity(r_pmmh(burnin+1:end));
plot(xr,fr,'r-');
tmp=get(gca,'ylim');
plot([r0 r0],tmp*2,'k-');
set(gca,'ylim',tmp);
hold off;
legend('PGAS','PMMH');
title('Empirical PDF of p(r | y_{1:T})')



