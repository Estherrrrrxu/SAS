
%--------------------------------------------------------------------------
function [xp,vp,Tpath,ft, w,ess] = cpf_as(y, ssmPar, Np, theta,sigma2,X)
% Conditional particle filter with ancestor sampling
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

Terms = ssmPar.Terms;  fnt   = ssmPar.fnt; 
Phi_n = @(Tn,tn) Tn*theta+fnt(tn);    % Phi_n dependes on time
p     = ssmPar.p;  q     = ssmPar.q;   mLag  = max(p,q);
stdObs= ssmPar.sigmaW;    %  std of the observatin noise
stdS  = sqrt(sigma2);     %  std of the noise in state model

conditioning = (nargin > 5);
T  = length(y);           % length of observations
xp = zeros(Np, T);        % Particles
a  = zeros(Np, T);        % Ancestor indices 
w  = zeros(Np, T);        % Weights
xp(:,1:mLag) = 0*randn(Np,mLag);   % Deterministic/random IC 
xp(Np,:)= X;     % Set the Np-th particle according to the conditioning
vp      = stdS*randn(Np,T);          % a noise sequence 
vp(:,1:mLag) = 0*vp(:,1:mLag);
Tpath   = zeros(Np,length(theta),T); % terms path
ess     = zeros(1,T); 

ft      = fnt(1:T); 
t=1; 
while t<=T
    if(t >mLag)
        ind = resampling(w(:,t-1));
        ind = ind(randperm(Np));        
        XX  = xp(ind,t-mLag:t-1); VV = vp(ind,t-mLag:t-1); % ind here- checked
        Tn  = Terms(XX,VV);         
        Tpath(:,:,t-1) = Tn;   
        xpred   = Phi_n(Tn,t-1);    % non-Markov model 
        xp(:,t) = xpred + vp(:,t);
        if(conditioning)
            xp(Np,t) = X(t); % Set the N:th particle according to the conditioning
            % Ancestor sampling
            wexp = -1/(2*stdS^2)*(X(t)-xpred).^2;
            wexp = wexp + logweights; 
            w_as = exp(wexp-max(wexp)); 

            w_as = w_as/sum(w_as);
            % ind(Np) = find(rand(1) <= cumsum(w_as),1,'first');
            temp = find(rand(1) <= cumsum(w_as),1,'first');
            if isempty(temp); keyboard; end
            
            ind(Np) = temp;
        end
        % Store the ancestor indices
        a(:,t) = ind;
    end
    % Compute importance weights
    ypred = fnObs_nl(xp(:,t));
    logweights = -1/(2*stdObs^2)*(y(t) - ypred).^2; % (up to an additive constant)
    const   = max(logweights); % Subtract the maximum value for numerical stability
    weights = exp(logweights-const);
    weights = weights/sum(weights); w(:,t)= weights; % Save the normalized weights
    ess(t)  = 1/(weights'* weights);  
    t = t+1;
end

% Generate the trajectories from ancestor indices
ind = a(:,T);
for t = T-1:-1:1+mLag
    xp(:,t) = xp(ind,t);
    vp(:,t)  = vp(ind,t);
    Tpath(:,:,t) = Tpath(ind,:,t);
    ind = a(ind,t);
end
xp(:,1:mLag) = xp(ind,1:mLag);
%  v     = v(:,mLag:end);  Tpath = Tpath(:,:,mLag:end);
vp (:,1:mLag) = vp(ind,1:mLag);
Tpath(:,1:mLag) = Tpath(ind,1:mLag);
end

function InitializeEnsemble(mLag,yobs,ssmPar,Np)
% initialize ensemble of particle filter
% DOES NOT seem worth it. 
func = 1; 

end
