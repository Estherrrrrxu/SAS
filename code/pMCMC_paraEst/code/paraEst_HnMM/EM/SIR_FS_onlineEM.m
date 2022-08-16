function [xp,vp,w,thetaN,stdS,ess] = SIR_FS_onlineEM(y,ssmPar,Np)
% SIR particle filter with forward only smoothing in online EM   
%% ======= TO DO next: 1. complete UpdateTheta; 2. Get compatible with estInd frame 
% Input:
%   y     - measurements
%   para  - parameters in state model
%   theta - estimated coefs in previous step
%   sigma2- variance of noise in state model
%   Np - number of particles
%   X  - conditioned particles - if not provided, un unconditional PF is run
% Output: 
%   xp v  - x,v particles
%   Tpath, Ttheta - paths of terms and terms*theta ~~ for theta regression
%   w     - weight of particles
%   essp  - effective sample size

tN  = length(y);           % length of observations
xp  = zeros(Np, tN);        % Particles
vp  = zeros(Np,tN);          % a noise sequence 
w  = zeros(Np, tN);        % Weights
L = length(ssmPar.theta0); 

thetaN = bsxfun(@plus,zeros(L,tN), ssmPar.theta0);  
stdS   = zeros(1,tN) + 0.5;     %  std of the noise in state model

Terms  = ssmPar.Terms;      fnt   = ssmPar.fnt; 
p      = ssmPar.p;          q     = ssmPar.q;   mLag  = max(p,q);
stdObs = ssmPar.sigmaW;    %  std of the observatin noise

xp(:,1:mLag) = 0*randn(Np,mLag);   % Deterministic/random IC
vp(:,1:mLag+1) = stdS(1)*randn(Np,mLag+1);   % Deterministic/random IC 
Termpath     = zeros(Np,L,tN); % terms path
ess          = zeros(1,tN); 
ft           = fnt(1:tN); 

TnAbc.A = zeros(L,L,Np); TnAbc.b = zeros(L,Np);  TnAbc.c = zeros(1,Np); 
gamma   = (1:tN).^(-0.6); 

for t=1:tN
    if(t >mLag)  % update the samples, and compute the weightT in Tn
        vp(:,t) = stdS(t-1)*randn(Np,1); 
        XX    = xp(:,t-mLag:t-1);     VV  = vp(:,t-mLag:t-1); 
        termn = Terms(XX,VV);          % size = (Np,L)  
        xpred   = termn * thetaN(:,t-1) + ft(t-1); 

        ind   = resampling(w(:,t-1));   ind = ind(randperm(Np));        
        Termpath(:,:,t-1) = termn(ind,:);           
        xp(:,t) = xpred(ind,:) + vp(:,t);
        % update TnAbc 
        Xnftn =  xp(:,t) -  ft(t-1); 
        TnAbc = UpdateT(xp(:,t),xpred,Xnftn,termn,logw,stdS(t-1),gamma(t),TnAbc); 
    end
    
    % Compute importance weights
    ypred = fnObs_nl(xp(:,t));
    logw    = -1/(2*stdObs^2)*(y(t) - ypred).^2; % (up to an additive constant)
    logw    = logw-max(logw); % Subtract the maximum value for numerical stability
    weights = exp(logw);
    weights = weights/sum(weights); w(:,t)= weights; % Save the normalized weights
    ess(t)  = 1/(weights'* weights);
   
    if (t >mLag+100)
        [theta1,sv] = update_theta(weights,TnAbc);
        thetaN(:,t) = theta1; stdS(t) = sv;
    end
end
end

function TnAbc = UpdateT(xp,xpred,Xnftn,termsn,logw,stdS,gamman,TnAbc)
% update the auxiliary term T and the surfficient stats S
% Input
%    Xnftn   - particles  Xn-ftn, Np x1  
%    termsn  -  terms, size = NpxL    L=length(theta)
%    logw    - log weight in the previous step n-1
%    TnAbc   - structure, TnAbc.A, .b. c
%    gamman  - gamma at time n

Np = length(xp(:,1)); 

for i=1:Np
    logP  = -1/(2*stdS^2)*(xp(i)-xpred).^2;   
    logP  = logP + logw;
    logP  = exp(logP-max(logP));
    weightT= logP /sum(logP);       % Np x 1
    tempA = 0; tempb = 0; tempc=0; 
    for j = 1:Np
        suffS = sufficentStat(Xnftn(i),termsn(j,:));
        tempA = tempA +( (1-gamman)*TnAbc.A(:,:,j) + gamman*suffS.A)*weightT(j); % NpxLxL 
        tempb = tempb +( (1-gamman)*TnAbc.b(:,j)  + gamman*suffS.b )*weightT(j); % NpxL
        tempc = tempc +( (1-gamman)*TnAbc.c(:,j)  + gamman*suffS.c )*weightT(j); % Npx1
    end
     TnAbc.A(:,:,i) = tempA;
     TnAbc.b(:,i)   = tempb;
     TnAbc.b(:,i)   = tempc;
end
end

function [theta,sv] = update_theta(wn,TnAbc)
% update theta at time n
% Input 
%    wn      - weight of particles Xn              Npx1   
%    TnAbc   - structure, TnAbc.A, .b. c
% Output
%    theta1  - estimator of coefficients A\b
%    sv      - estimator of variance of state noise
% 
Np = length(wn); 

for i =1:Np
    A = wn(i)* TnAbc.A(:,:,i);
    b = wn(i)* TnAbc.b(:,i);  % Lx1
    c = wn(i)* TnAbc.c(:,i);
end
theta = A\b;    % Lx1
sv     = c -2* b'*theta+ theta'*A*theta; 
% |xnftn- theta*terms|^2 = c - 2*b*theta1 + theta1'*A*theta1;    

end

function suffS= sufficentStat(Xnftn,termsn)
% compute the sufficient statistics
% Input:
%   Xnftn   - a particle,  Xn-ftn, 1x1   
%   termsn  - terms,               1xL    L=length(theta) 

suffS.A = termsn'*termsn;     % LxL    % since A does not depend on Xnftn 
suffS.b = termsn'*Xnftn;      % Lx1
suffS.c = Xnftn.^2;           % 1x1
end

