function [prior,stdF] = setPrior(prior,nldeg)
%% set prior for the Bayesian inference
prior.statebounds = [0.5,1.5];      % lower and upper bounds of state

if prior.regul ==0; fprintf('Method: standard Bayeisan inference,    ');
else;               fprintf('Method: Reluarized likelihood Bayes,  '); end
if prior.flag == 0;     fprintf('  Gaussian prior\n');
elseif prior.flag ==2; fprintf('  Uniform prior\n'); end

% -- the number of parameters also determines the term in the nl_fn.  ***
% -- size of mu:    1xK for nl_fn, otherwise Kx1 (e.g. prior, samples);
[thetatrue,stdF, thetaLo, thetaUp, thetaStd] = theta_alpha(nldeg);    
% theta_alpha;     % load thetatrue, lower and upperbound, thetaStd  
    thetaTrue = thetatrue; 
            prior.lb  = thetaLo';     prior.ub  = thetaUp';
switch prior.flag
    case 0    %  % Gaussian prior  
        prior.mu  = thetaTrue';   prior.std = 1*thetaStd';   % 1xK vector
        temp      = prior.mu + randn(size(prior.mu)).*prior.std;
        prior.lb6std  = prior.mu - 6*prior.std; prior.ub6std = prior.mu + 6*prior.std; 
        prior.initialguess = temp';   % initialguess for sampler; 1xK for nl_fn
    case 2    %  % Uniform prior
        temp      = prior.lb + rand(size(prior.lb)).*(prior.ub-prior.lb); 
        prior.initialguess = temp';   % 1xK for nl_fn,
        prior.mu  = thetaTrue';  % prior.mu center
    case 10      % Gaussian prior  -- original
        thetaTrue = [1,-0.1,0.2,-0.1,-1];
        % thetaTrue = [0.1,-1,0.4,-1,0.2]; % original
        prior.mu  = thetaTrue';
        prior.std = [0.1; 0.04; 0.05; 0.03; 0.1].^2;  % sigma_i^2
        
    case 12     % uniform original
        thetaTrue = [1,-0.1,0.2,-0.1,-1]*0.9;
        prior.mu  = thetaTrue';  % Here prior.mu used to pass the true value.
        prior.lb  = [0.8; -0.2;0; -0.2; -1.2];
        prior.ub  = [1.1; 0.1; 0.4; 0; -0.8];
    case 11     % Lognormal
        % prior [log(N(m0,s0^2)),N(m123,s123^2), -logN(m4,s4^2)]
        prior.mu  = [0;-0.1;0.2;-0.1;0];   % mu_i in the prior: Kx1 vector
        prior.std = [1; 0.1; 0.1; 0.1; 2].^2;  % sigma_i^2
        temp = prior.mu;    % use prior mean as the true parameter
        thetaTrue = [exp(temp(1)), temp(2:4), -exp(temp(5))]; % 1x5 vector
end

clear thetaLo thetaUp thetaStd  thetaTrue thetatrue temp; 
