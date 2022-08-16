function [theta,sV2] = samplePar_MLE(X,V,T,ft,ssmPar)
% sample the parameters from posterior distribution; 
% assuming uniform/Gaussian prior
% Input
%   X,V,T   -- a trajectory of X, V and Terms
%   ssmPar  - includes the prior and state space model
% Output
%  sample of theta and variance of the noise in state model

%% the MLE from X and V:  A * theta =b
Xft  = X- ft;  
mLag = max(ssmPar.p, ssmPar.q); ind  = mLag:length(X);  
X    = X(ind); V = V(ind); T = T(:,ind); Xft = Xft(ind); 

A      = T * T';   b  = T * Xft'; % theta = A\b;       
estInd = ssmPar.thetaInd; knownInd = setdiff(1:4,estInd); 
b     =  b(estInd)- A(estInd,knownInd)*ssmPar.thetaTrue(knownInd);
A     = A(estInd,estInd); 
theta = A\b; 
sV2   = var(V); 

inflation = 1.1; sV2 = sV2*inflation;  

% fprintf('\n True (top row) and MLE estimator: coefs, sigmaV\n')
% estInd = ssmPar.thetaInd; 
% disp([ssmPar.thetaTrue(estInd)', ssmPar.sigmaV; theta', sV2]); 
trueThsv = [ssmPar.thetaTrue(estInd)', ssmPar.sigmaV]; 
mleThsv  = [ theta', sV2];
fprintf('\n True:  '); fprintf(' %4.2f  ', trueThsv); 
fprintf('\n MLE :  '); fprintf(' %4.2f  ', mleThsv); 
fprintf('\n         coefs,    sigmaV ----  Progress:    ');

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
estInd = ssmPar.thetaInd; knownInd = setdiff(1:4,estInd); 
b     =  b(estInd)- A(estInd,knownInd)*ssmPar.thetaTrue(knownInd);
A     = A(estInd,estInd); 
theta = A\b; 
sV    = var(Vtrue); 

end