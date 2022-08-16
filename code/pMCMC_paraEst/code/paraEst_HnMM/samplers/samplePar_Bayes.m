function [theta,sV2, theta_mle,sV_mle] = samplePar_Bayes(X,V,tms,ft,ssmPar, sampleVar)
% sample the parameters from posterior distribution; 
% assuming uniform/Gaussian prior
% Input
%   X,V tms   -- a trajectory of X, V and Terms
%            X(1:tN), V(1:tN)
%   ssmPar  - includes the prior and state space model
% Output
%  sample of theta and variance of the noise in state model

%% Bayesian estimator
% make the data to be of the same length  
 
tN   = length(X); 
ind  = floor(tN/2):tN-1;     % use only half of the estimated path
V    = V(ind);               % V used in variance estimation
Xft  = X(ind+1)- ft(ind);    % Xft = X_{n+1} - ft(n) = tms*theta 
tms  = tms(:,ind);           % terms stacked  

% sample the variance sigmaV2
if exist('sampleVar','var') && sampleVar ==1
    % sample sV from inverse Gamma distribution: 1./gamrnd;
    beta   = sum(V.^2)/2 + ssmPar.priorsVb;
    alpha  = (length(V)-1)/2 + ssmPar.priorsVa;
    sV2    = 1/gamrnd(alpha,1/beta); %
else
    sV2    =  ssmPar.sigmaV.^2;
end

inflation = 1.0; sV2 = sV2*inflation;  

% the likelihood: it provide a Gaussian for theta: mu1, cov1
% -- a byproduct: the MLE is mu1, and sV_mle is the variance of V
estInd   = ssmPar.thetaInd;         % index of the parameters to be estimated
knownInd = setdiff(ssmPar.allInd,estInd); 
A     = tms * tms';   b  = tms * Xft'; % theta = A\b; 
b     =  b(estInd)- A(estInd,knownInd)*ssmPar.thetaTrue(knownInd);
A     = A(estInd,estInd); 
theta_mle = A\b; 
sV_mle    = var(V); 
cov1inv   = A/sV2;       mu1 = theta_mle;  

if ssmPar.prior ==1     %%%%  Gaussian prior   
    % update mean and covariance of theta
    cov0inv   = diag(1./ssmPar.priorstd(estInd).^2);
    mu0       = ssmPar.priormu(estInd);
    covInv    = cov0inv + cov1inv ;
    mu        = covInv \(cov1inv*mu1 + cov0inv*mu0);
    cinvchol   = chol(covInv);
    % sample theta from the posterior
    theta     = mu + cinvchol\randn(length(mu1),1);
elseif ssmPar.prior ==2   %%%%  Uniform prior ---------------- BUG: temp gets to INF -------=======
    c1invchol = chol(cov1inv);  % cov1inv = C*C'  
    lb    = c1invchol\(ssmPar.priorlb(estInd)- mu1); 
    ub    = c1invchol\(ssmPar.priorub(estInd) -mu1);
    temp  = trandn(lb,ub);
    theta = mu1+ c1invchol*temp;
    if isnan(theta)
        fprintf('\n Imaginal theta! Check cinvchol. \n');   keyboard;
    end
end
trueThsv = [ssmPar.thetaTrue(estInd)', ssmPar.sigmaV]; 
mleThsv  = [ theta_mle', sV_mle]; 
bayesThsv= [theta', sV2]; 
fprintf('\n True:  '); fprintf(' %4.2f  ', trueThsv); 
fprintf('\n MLE :  '); fprintf(' %4.2f  ', mleThsv); 
fprintf('\n Bayes: '); fprintf(' %4.2f  ', bayesThsv); 
fprintf('\n         coefs,    sigmaV ----  Progress:    '); 
end 

function [theta,sV] = MLE(Xtrue,Vtrue, ssmPar)
% Estimate the parameter by MLE from the estimated trajectory X and V
% not used to save computational time, because it computes the path again 

tN    = length(Xtrue);    Terms = ssmPar.Terms; fnt  = ssmPar.fnt; 
mLag  = max(ssmPar.p, ssmPar.q); 
tms = zeros(length(ssmPar.thetaTrue),tN-mLag); 
Xft = zeros(1,tN-mLag); 
for t = mLag+1:tN-1
    XX = Xtrue(:,t-mLag+1:t); 
    VV = Vtrue(:,t-mLag+1:t);
    tms(:,t) = Terms(XX,VV);
    Xft(:,t) = Xtrue(t+1) - fnt(t);  
end

%% the MLE: Ax = b 
A      = tms * tms';   b  = tms * Xft'; % theta = A\b;       
estInd = ssmPar.thetaInd; knownInd = setdiff(ssmPar.allInd,estInd); 
b     =  b(estInd)- A(estInd,knownInd)*ssmPar.thetaTrue(knownInd);
A     = A(estInd,estInd); 
theta = A\b; 
sV    = var(Vtrue); 

end