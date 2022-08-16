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
% The script generates a batch of data y_{1:T} from the the standard
% nonlinear time series model,
%
%   x_{t+1} = 0.5*xt + b*x_{t-1}+ 25*x_t/(1+x_t^2)+8*cos(1.2*t)+c*v_{t-1}+ v_t,
%   y_t = 0.05*x_t^2 + e_t,
%
% with v_t ~ N(0,q) and e_t ~ N(0,r). The process noise and measurement
% noise variances (q,r) are treated as unkown parameters with inverse Gamma
% priors. The PGAS and PMMH algorithms are then executed independently to
% find the posterior parameter distribution p(q, r | y_{1:T}).
%
% Original  Fredrik Lindsten;    Linköping, 2013-02-19    lindsten@isy.liu.se
% Modified: Fei Lu 2018-4-3 

%% PGAS  ==========================================================
% Set up parameter of PGAS
Np = 5;                  % Number of particles used in PGAS
numMCMC = 1000;          % Number of iterations in the MCMC samplers
burnin  = 100;            % Number of interations to burn
plotOn  = 1;            % Plot intermediate sample paths?
para.priorsVa = 0.1;   para.priorsVb =0.1;


%% Settings of the state space model
a1= 0.9; a2= -0.20; b1=25; d1=0.8; coefs = [a1;a2;b1;d1]; p=2;q=1;  
fnt =@(n)8*cos(1.2*(n-1) );
para.coefs  = coefs;
para.fnt    = fnt;   
para.sigmaW = 1;          % std of the observation noise
para.sigmaV = 1;        % std of the noise in the state model
para.N   = 100;       % number of time steps, = # of observations
para.p   = p; 
para.q   = q;
meanFlag = 0; 

% % % Get Phi
Terms = @(XX,VV) terms_narma(XX,VV, p,q,meanFlag);    
Phi_n = @(XX,VV,tn) Terms(XX,VV)*coefs+fnt(tn);    % Phi_n dependes on time
para.Terms = Terms;

%% Generate data
[Xtrue,obs] = generate_data(para,Phi_n,plotOn);


%% Estimate the states by PGAS
fprintf('Running PGAS (N=%i). Progress: ',Np); tic;
[x_pgas] = pgas_state(numMCMC, obs, Np, para,Phi_n); 
timeelapsed = toc;
fprintf(' Elapsed time: %2.2f sec.\n',timeelapsed);


%% Plot the results
close all;

figure
plot(x_pgas(1,:), 'b','linewidth',1); hold on
plot(x_pgas(500,:),'c-+','linewidth',1 ); hold on
plot(x_pgas(1000,:),'g-x','linewidth',1  ); hold on
h2 = plot(Xtrue, 'k:','linewidth',2); 
legend('t=1','t=500','t=1000','true')
title('Trajectory ensemble')

% % % ---------------------------------------
% state estimation results 
figure(9);   % plot the marginal densities
tt= 1:10:para.N; tt = [tt,para.N]; 
ttN = length(tt);
xArray = zeros(ttN,100); 
fArray = zeros(ttN,100);
titl   = strings(ttN,1);
for k =1: ttN
[fr,xr] = ksdensity(x_pgas(burnin+1:end,tt(k)));
xArray(k,:) = xr; fArray(k,:) = fr; 
plot(xr,fr,'b-');  
tmp = get(gca,'ylim'); 
set(gca,'ylim',tmp);  legend('PGAS');
titl(k) = sprintf('p(x(t) | y_{1:%3g}) with t=%3g', para.N,tt(k));
title(titl(k))
drawnow;   pause(0.2)
axis tight;
end
AnimateGif(xArray,fArray,titl); % % Animated gif


figure(7);   % plot the mean and std of trajectories
xmean = mean(x_pgas(burnin+1:end,:)',2); 
xstd  = std(x_pgas(burnin+1:end,:)); xstd = xstd';
plot(xmean); hold on;
plot(xmean+xstd,'r--'); 
plot(xmean-xstd,'r--');
plot(Xtrue, 'k:','linewidth',2); 
legend('MeanEst','true','mean+std','mean-std')
title('Mean and Std of trajecotry samples')

figure(8);clf  % plot the trajectory ensemble
plot(x_pgas(burnin+1:end,:)','g'); hold on
h1 = plot(xmean,'b','linewidth',1);
h2 = plot(Xtrue, 'k:','linewidth',2); 
legend([h1,h2],{'ensemble mean','true'})
title('Trajectory ensemble')





