% Input the settings of the state space model and the priors
% The state space model: 
%   x_{t+1} = a_1x_t + a_2x_{t-1}+ b1 x_t/(1+x_t^2)+8\cos(1.2t)+d_1v_{t-1}+ v_t,
%   y_t = 0.05*x_t^2 + W_t,
%
% The priors: 
%    - uniform or Gaussian for the coefficients, 
%    - Inverse Gamma for the variance of noise in the state model

 rng(12); % the seed for generating true parameters

%% set the prior of the parameters: coefs, variance
% these coefficients shall be drawn from prior later. 
if ssmPar.prior ==1    %%% === Gaussian prior of parameters
    mu  = [0.7;-0.2;25; 0.75]; % [a1; a2; b1; d1]  % A COLUMN vector
          % a1,a2 linear stability: roots of 1-a1*z - a2*z^2 outside of unit ball.
    std = [0.1; 0.05; 2; 0.05];  % sigma_i
    thetaTrue = mu + 0*randn(size(mu)).*std;  
    ssmPar.priormu   = mu;           ssmPar.priorstd = std; 
    ssmPar.thetaTrue = thetaTrue; 
    ssmPar.theta0    = mu + randn(size(mu)).*std; % initial guess for sampling
elseif ssmPar.prior ==2     %%% === Uniform prior
    lb  = [0.6; -0; 20; 0];    ssmPar.priorlb = lb; 
    ub  = [0.9; -0; 30; 0];   ssmPar.priorub = ub;
    thetaTrue = rand(size(lb)).*(ub-lb) + lb;  
    ssmPar.thetaTrue = thetaTrue;  
    ssmPar.theta0    = rand(size(lb)).*(ub-lb)+lb; % initial guess for sampling
end
fprintf('True para [a1 a2 b1 d1]:'); fprintf(' %3.2f ',thetaTrue); fprintf('\n'); 

% linear stability 
p = [1,-thetaTrue(1:2)']; 
fprintf('Roots of 1-a1*z-a2*z^2 (should be outside of unit circle) \n'); 
root = roots(fliplr(p));   disp(root); 

alpha  = 2; beta= 1;       sigma2 = 1/gamrnd(alpha,1/beta);   
ssmPar.priorsVa = alpha;   ssmPar.priorsVb = beta;  % Inverse Gamma for sV
ssmPar.sigmaV   = 1+ 0*sqrt(sigma2);  
fprintf('True std of the model noise: %2.1f \n', ssmPar.sigmaV ); 

%% Settings of the state space model 
p=2; q=1;   
fnt =@(n)0*cos(1.2*(n-1) ); 

ssmPar.fnt    = fnt;   
ssmPar.sigmaW = .5;     % std of the observation noise
ssmPar.sigmaV = 1;      % std of the noise in the state model
ssmPar.p   = p; 
ssmPar.q   = q;

% % % Get Phi
meanFlag = 0; 
Terms    = @(XX,VV) terms_narma(XX,VV, p,q,meanFlag);    
Phi_n    = @(XX,VV,tn) Terms(XX,VV)*thetaTrue+fnt(tn);   % Phi_n dependes on time
ssmPar.Terms = Terms; 
ssmPar.Phi_n = Phi_n;

clearvars -except ssmPar; 


