function [x,w ] = pf_NodesSIR(obs,obsPar,femPar,pmcmcPar,theta0)
% Particle filter, using SIR
% Input:
%   obs  - measurements: size = [# of time intervals, # regions]
%                            = [length(t0t1(:,1)), length(regions(1,:))]
%   obs    - measurements: size = [# of time intervals, # regions]
%                            = [length(t0t1(:,1)), length(regions(1,:))]
%   obsPar, femPar, pmcmcpar - parameters for obs, fem and pmcmc
%   theta0  -  value of theta 
% Last updated: 2019-1-12
%
% % State model in Gaussian transition format:
%             U_n = bn + sqrt(dt)*stdF*ABmat\Prec_chol\N(0,Id)
%                 = bn + N(0,C_U),    
%  where   bn  = udet from forward_ensemble.m
%          C_U = dt*stdF^2*Cmat*Cmat^T, with Cmat^{-1} = ABmat*Prec_chol
% % Observation model
%             Y_n = H U_n + V_n


forward   = femPar.forward;        dt    = femPar.dt; 
forward1  = @(U0,dt,tn) forward(U0,dt,tn,theta0); 

varObs    = obsPar.stdObs^2;       nodes = obsPar.nodes; 
% initialize state sample (particle) traj's 
Dx        = obsPar.stateDim;  Np  = pmcmcPar.Np;   
T         = obsPar.tN +1;   % length of state traj; obs(tt) is observation of x(tt+1) 
x         = zeros(Dx,T,Np);         % Particles
x(:,1,:)  = 1+0.01*randn(Dx,Np);    %  initial ensemble   
w   = zeros(Np, T);  w(:,1) = ones(Np,1)/Np; % particle Weights; same length as x  
ttN = length(obs(1,:));   % number of SMC time steps = obs Times 

for tt =1:ttN     % % ======  NOTE: obs(tt) is observation of x(tt+1) 
    t0 = tt; t1 = t0 +1; steps = 1;  % for extension to t0t1. Observe every step here
    xt0 = reshape(x(:,t0,:), Dx,Np);  
    ind = myResampling(w(:,tt));     ind = ind(randperm(Np));
    if tt ==1; ind = 1:Np; end
    [xpred,~] = forward_ensemble(forward1,xt0,dt,steps,ind);  % size(xpred)=[Dx,steps+1,Np]                      
     x(:,t0+1:t1,:) = xpred(:,2:end,:);  
    % Compute importance weights:
    UU      = reshape(x(:,t1,:),Dx,Np);    
    ypred   = UU(nodes,:);                % size=[rN,Np]
    logweights = -1/(2*varObs)*sum( (obs(:,tt) - ypred).^2,1); 
    const   = max(logweights); % Subtract the maximum value for numerical stability
    weights = exp(logweights-const);
    w(:,tt+1) = weights/sum(weights); % Save the normalized weights
end

end

