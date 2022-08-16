function [theta,sV2,surffS] = SAEM(X,V,T,ft,ssmPar,gammat,surffS)
% SAEM for pgag to estimate MLE: using only the reference trajectory
% Input
%   X,V,T   -- an ensemble of trajectories of X, V and Terms
%   ssmPar  - includes the prior and state space model
% Output
%  MLE of theta and variance of the noise in state model

%% the MLE from X and V:  A * theta =b
Xft  = X- ft;  
mLag = max(ssmPar.p, ssmPar.q); ind  = mLag:length(X);  
X    = X(ind); V = V(ind); T = T(:,ind); Xft = Xft(ind); 

A      = T * T';   b  = T * Xft'; % theta = A\b;       
estInd = ssmPar.thetaInd; knownInd = setdiff(ssmPar.allInd,estInd); 
b     =  b(estInd)- A(estInd,knownInd)*ssmPar.thetaTrue(knownInd);
A     = A(estInd,estInd); 
A     = (1-gammat)*surffS.A + gammat*A; 
b     = (1-gammat)*surffS.b + gammat*b; 
theta = A\b; 

%% estimate the variance: use V or Vest: Vest leads to large oscilations
estV = 0; 
if estV ==1
    thetaEst = ssmPar.thetaTrue; thetaEst(estInd) = theta;
    Vest  = Xft - thetaEst'*T;
    sV2 = var(Vest);    % large values: >50, while sigmaV =1.  
else
    sV2   = var(V);     % tend to under-estimate sigmaV 
end

sV2   = (1-gammat)*surffS.sV2 + gammat*sV2;

% fprintf('\n True (top row) and MLE estimator: coefs, sigmaV\n')
% estInd = ssmPar.thetaInd; 
% disp([ssmPar.thetaTrue(estInd)', ssmPar.sigmaV; theta', sV2]); 
trueThsv = [ssmPar.thetaTrue(estInd)', ssmPar.sigmaV]; 
mleThsv  = [ theta', sV2];
fprintf('\n True:  '); fprintf(' %4.2f  ', trueThsv); 
fprintf('\n MLE :  '); fprintf(' %4.2f  ', mleThsv); 
fprintf('\n         coefs,    sigmaV ----  Progress:    ');

surffS.A   = A; 
surffS.b   = b; 
surffS.sV2 = sV2; 
end
