function [x,w,sFlag] = cpf_asNodes_SIR(obs,obsPar,femPar,pmcmcPar,theta0,X1)
% Conditional particle filter with ancestor sampling, using SIR
% -- requires observation every time step; 
% Input:
%   obs    - measurements: size = [# of time intervals, # regions]
%                            = [length(t0t1(:,1)), length(regions(1,:))]
%   obsPar, femPar, pmcmcpar - parameters for obs, fem and pmcmc
%   theta0  -  value of theta 
%   X1      - conditioned particles - must be provided
%   % previous:  t0t1 - start and end of time intervals  ttNx2; connected (for SMC)
%            regions- nodes in regions, size= [ttN*rN, eN]
% Last updated: 2019-1-12
%
% % State model in Gaussian transition format:
%             U_n = bn + sqrt(dt)*stdF*ABmat\Prec_chol\N(0,Id)
%                 = bn + N(0,C_U),    
%  where   bn  = udet from forward_ensemble.m
%          C_U = dt*stdF^2*Cmat*Cmat^T, with Cmat^{-1} = ABmat*Prec_chol
% % Observation model
%             Y_n = H U_n + V_n

pgas = pmcmcPar.pgas;     %  1 for ancester sampling; 0 for cpf without AS

forward1step   = femPar.forward1step;        dt    = femPar.dt; 
forward1  = @(U0) forward1step(U0,theta0); 
chol_PAB  = femPar.Prec_chol*femPar.ABmat; 
stdF      = femPar.stdF;           stdF2 = stdF.^2; 
varObs    = obsPar.stdObs^2;       nodes = obsPar.nodes; 
% initialize state sample (particle) traj's 
Dx        = obsPar.stateDim;  Np  = pmcmcPar.Np;   
T         = obsPar.tN +1;   % length of state traj; obs(tt) is observation of x(tt+1) 
x         = zeros(Dx,T,Np);         % Particles
x(:,1,:)  = 1+0.01*randn(Dx,Np);    %  initial ensemble of samples  

X_Np      = reshape(X1,[Dx,T]);  % size(X1)=[ Dx,T,1]->>[Dx T]. Ref path
x(:,:,Np) = X1(:,:,1);    % Set the Np:th particle as the reference traj
w   = zeros(Np, T);  w(:,1) = ones(Np,1)/Np; % particle Weights; same length as x  
ttN = length(obs(1,:));   % number of SMC time steps = obs Times 
a   = zeros(Np, ttN);     % Ancestor indices

sFlag = 0 ;       % indicator of singular ancestor weightstemp 
for tt =1:ttN     % % ======  NOTE: obs(tt) is observation of x(tt+1) 
    t0 = tt; t1 = t0 +1;  % for extension to t0t1. Observe every step here
    xt0 = reshape(x(:,t0,:), Dx,Np);  
    ind = myResampling(w(:,tt));     ind = ind(randperm(Np));
    ind = ind(1:Np-1);  % NEW: sample Np-1 particles 
    if tt ==1; ind = 1:Np-1; end 
    [u1det,xpred] = forward_ensemble1step(forward1,xt0,ind); 
                  % size(xpred)=[Dx,Np-1]   % size(u1det) = [Dx,Np]  
                  % u1det is the 1-step deterministic forward of the particles                     
    x(:,t1,1:Np-1) = xpred;   % NEW: Np-th particle not chnaged here   
    % Compute importance weights:
    UU      = reshape(x(:,t1,:),Dx,Np);    
    ypred   = UU(nodes,:);                % size=[rN,Np]
    logweights = -1/(2*varObs)*sum( (obs(:,tt) - ypred).^2,1); 
    const   = max(logweights); % Subtract the maximum value for numerical stability
    weights = exp(logweights-const);
    w(:,tt+1) = weights/sum(weights); % Save the normalized weights
   
    if tt>1 && pgas     % Ancestor sampling for the Np-th particle (reference traj)
        X_Np1= X_Np(:,t0+1);    
        wexp = multiPriorWt(X_Np1,u1det,chol_PAB,dt,stdF2);
        
        wexp = wexp + logweights'; % use the exponents to avoid singular
        w_as = exp(wexp- max(wexp));           w_as = w_as/sum(w_as);
        temp = find(rand(1) <= cumsum(w_as),1,'first');
        if isempty(temp); sFlag=1; fprintf('Singular ancestor weights!\n'); 
            temp= Np; end   % set ancester of the Np-th paritcle as itself 
        ind(Np) = temp;  
        a(:,tt) = ind;   % ancestor indices -- starting from tt=2
    end
end

% PGAS: Generate the trajectories from ancestor indices: 
if pgas;       ind = a(:,ttN);   
    for tt = ttN-1:-1:1 
        t0= tt; t1 = t0+1; 
        if tt>1;   x(:,t0+1:t1,:) = x(:,t0+1:t1,ind);  % avoid overlap
        else;      x(:,t0:t1,:)   = x(:,t0:t1,ind);            end 
        ind = a(ind,tt);
    end
end
end


%-------------------------------------------------------------------
function  wexp = multiPriorWt(X_Np1,xpred_det,chol_PAB ,dt,stdF2)  % UPDATED
% compute likelihood weight in ancester sampling  >> only one step needed
%   X_Np1    - reference trajectory at t0+1 (Corrected from t0:t1) size = [Dx,1]
%   xpred_det- predicted by deterministic forward: size = [Dx,Np]
%   chol_PAB  - Cholesky facotorization of the covariance matrix
% Formular
%      U_{n+1} = pred_det + M2^{-1} N(0, dt M) 

[~,Np] = size(xpred_det);
wexp     = zeros(Np,1);
for nn = 1:Np
    diff  = X_Np1 - xpred_det(:,nn);  
    pdiff = chol_PAB * diff;    % --Corrected 18/6/19, see notes FEM_SEB.tex ==== 
    wexp(nn) = - sum(pdiff.^2)/(2*stdF2*dt);  
    %NILS: Wrong sign and need to divide by 2, i.e. wexp(nn) = - sum(pdiff.^2)/(2*stdF2*dt)
end
end

