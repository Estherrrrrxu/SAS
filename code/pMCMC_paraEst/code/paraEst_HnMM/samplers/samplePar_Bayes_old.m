function [theta,sV2] = samplePar_Bayes(X,V,T,ft,ssmPar, sampleVar)
% sample the parameters from posterior distribution; 
% assuming uniform/Gaussian prior
% Input
%   X,V,T   -- a trajectory of X, V and Terms
%   ssmPar  - includes the prior and state space model
% Output
%  sample of theta and variance of the noise in state model

%% the MLE from X and V 
[thetaMLE,sVMLE] = MLE(X,V, ssmPar); 
fprintf('\n True (top row) and MLE estimator: coefs, sigmaV\n')
estInd = ssmPar.thetaInd; 
disp([ssmPar.thetaTrue(estInd)', ssmPar.sigmaV; thetaMLE', sVMLE]); 

%% Bayesian estimator
% make the data to be of the same length 
mLag = max(ssmPar.p, ssmPar.q); ind  = mLag:length(X);  
X    = X(ind); V = V(ind); T = T(:,ind); Xft = X- ft(ind); 

% sample the variance sigmaV2
%  invSigmaV = diag(1./ ssmPar.sigmaV.^2);
if exist('sampleVar','var') && sampleVar ==1
    % sample sV from inverse Gamma distribution: 1./gamrnd;
    beta   = sum(V.^2)/2 + ssmPar.priorsVb;
    alpha  = (length(V)-2)/2 + ssmPar.priorsVa;
    sV2    = 1/gamrnd(alpha,1/beta); %
else
    sV2    =  ssmPar.sigmaV.^2;
end

estInd   = ssmPar.thetaInd;         % index of the parameters to be estimated
knownInd = setdiff(ssmPar.allInd,estInd); 
temp     = ssmPar.thetaTrue(knownInd)' * T(knownInd,:);
Xft      = Xft - temp;
% sample theta from the likelihood: a Gaussian distribution.
c1inv = T(estInd,:) *T(estInd,:)' / ( sV2);  
mu1   = T(estInd,1:end-1)*Xft(2:end)'; 
mu1   = c1inv\ (mu1/ ( sV2)) ;
% cov1inv = chol(c1inv); 
%%% === replace the above: to drop the singular eigenvale of c1inv. 
%%% --- 6/28: this would lead to imaginary number in uniform prior
    tol =1e-1;  % tolerance in decompostion of covariance singular values
    [U,S,~] = svd(c1inv);         S    = sqrt( real(diag(S)) );
    ii      = find(abs(S)>tol);   Sinv = 0*S + 1/tol;  
    Sinv(ii)= 1./ S(ii);          cov1 = U'*diag(Sinv)*U; 
    cov1inv = U'*diag(1./Sinv)*U; 
    mu1 = cov1*mu1 ; % c1inv\ mu1;  
    

if ssmPar.prior ==1     %%%%  Gaussian prior   
    % update mean and covariance of theta
    cov0inv   = diag(1./ssmPar.priorstd(estInd).^2);
    mu0       = ssmPar.priormu(estInd);
    covInv    = cov0inv + cov1inv ;
    mu        = covInv \(cov1inv*mu1 + cov0inv*mu0);
    cinvchol   = chol(covInv);
    % sample theta from the posterior
    theta     = mu + cinvchol\randn(length(mu1),1);
elseif ssmPar.prior ==2   %%%%  Uniform prior ---------------- BUG: temp gets to INF -------=================
    c1invchol = chol(cov1inv); 
    lb   = c1invchol\(ssmPar.priorlb(estInd)- mu1); 
    ub   = c1invchol\(ssmPar.priorub(estInd) -mu1);
    temp = trandn(lb,ub);
    theta = mu1+ c1invchol*temp;
    if isnan(theta)
        fprintf('\n Imaginal theta! Check cinvchol. \n');   keyboard;
    end
end

% fprintf('Bayesian Estimtor:\n')  disp([theta', sV2]); 
end

function [theta,sV] = MLE(Xtrue,Vtrue, ssmPar)
% Estimate the parameter by MLE from the estimated trajectory X and V

tN    = ssmPar.tN;  Terms = ssmPar.Terms; fnt  = ssmPar.fnt; 
mLag  = max(ssmPar.p, ssmPar.q); 
tms = zeros(length(ssmPar.thetaTrue),tN-mLag); 
Xft = zeros(1,tN-mLag); 
for t = mLag+1:tN
    XX = Xtrue(:,t-mLag+1:t); 
    VV = Vtrue(:,t-mLag+1:t);
    tms(:,t) = Terms(XX,VV);
    Xft(:,t) = Xtrue(t) - fnt(t);  
end

%% the MLE: Ax = b 
A      = tms * tms';   b  = tms * Xft'; % theta = A\b;       
estInd = ssmPar.thetaInd; knownInd = setdiff(ssmPar.allInd,estInd); 
b     =  b(estInd)- A(estInd,knownInd)*ssmPar.thetaTrue(knownInd);
A     = A(estInd,estInd); 
theta = A\b; 
sV    = var(Vtrue); 

end