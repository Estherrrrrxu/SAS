function [x,w ] = pf_NodesIPF_statePrior(obs,obsPar,femPar,pmcmcPar,theta0)
% Particle filter, using IPF  
% + added: regularize by statePrior from obs

% Input:
%   obs  - measurements: size = [# of time intervals, # regions]
%                            = [length(t0t1(:,1)), length(regions(1,:))]
%   obsPar, femPar, pmcmcpar - parameters for obs, fem and pmcmc
%   theta0  -  value of theta 
% Last updated: 2019-1-14
%
% % State model in Gaussian transition format:
%             U_n = bn + sqrt(dt)*stdF*ABmat\Prec_chol\N(0,Id)
%                 = bn + N(0,C_U),    
%  where   bn  = udet from forward_ensemble.m
%          C_U = dt*stdF^2*Cmat*Cmat^T, with Cmat^{-1} = ABmat*Prec_chol
% % Observation model
%             Y_n = H U_n + V_n


obsPar = statePrior(obsPar,obs); % obtain state prior (mean&std) from obs

forward1step   = femPar.forward1step;        
forward1  = @(U0) forward1step(U0,theta0); 

% initialize state sample (particle) traj's 
Dx        = obsPar.stateDim;  Np  = pmcmcPar.Np;   
T         = obsPar.tN +1;   % length of state traj; obs(tt) is observation of x(tt+1) 
x         = 1+ zeros(Dx,T,Np);       % Particles
%  initial ensemble of samples: use obs
nodes     = obsPar.nodes;    nodesrest = setdiff(1:Dx,nodes); 
x(nodes,1,:)     = obs(:,1) + 0.01*randn(length(nodes),Np);
x(nodesrest,1,:) = 1 +  0.01*randn(length(nodesrest),Np);    


w   = zeros(Np, T);  w(:,1) = ones(Np,1)/Np; % particle Weights; same length as x  
ttN = length(obs(1,:));   % number of SMC time steps = obs Times 

for tt =1:ttN     % % ======  NOTE: obs(tt) is observation of x(tt+1) 
    t0 = tt; t1 = t0 +1;  % Observe every step here
    % resamling first: 
    ind = myResampling(w(:,tt));     ind = ind(randperm(Np));
    if tt ==1; ind = 1:Np; end        
    x(:,1:t0,:) =  x(:,1:t0,ind);   
    xt0         = reshape(x(:,t0,:), Dx,Np);  
    [sample, wt] = Optimal_ImpS(forward1,xt0,obs(:,tt),pmcmcPar,obsPar); 
    x(:,t1,:) = sample;  
    % if mod(tt,10)==9; keyboard; end
    w(:,tt+1)      = wt'; 
    % figure(17); clf; titl ='SMC: IPF. Node 5'; plotTrajpf(x,5,titl); pause(0.05);
end

end


function [sample,wt] = Optimal_ImpS(forward1,xt0,obs1step,pmcmcPar,obsPar)
% Implicit sampling = optimal sampling: 
%  - forward particles from the exact 1-step posterior distribution 
%  - the weights depend on xt0; output noramlized weights
%  - xt1ref is from the reference traj; not updated, but used in weights computation
% 
% % State model in Gaussian transition format:
%             U_n = bn + sqrt(dt)*stdF*ABmat\Prec_chol\N(0,Id)
%                 = bn + N(0,C_U),    
%  where   bn  = udet from forward_ensemble.m
%          C_U = dt*stdF^2*Cmat*Cmat^T, with Cmat^{-1} = ABmat*Prec_chol
% % Observation model
%             Y_n = H U_n + V_n
% % In optimal importance/Implicit sampling, sample of U_n is dran from
% Gaussian with mean and covariance
%             mean = bn + C_U *H'*C_Y^{-1} *(y_n-H*bn)
%             Cov  = C_U - C_U*H'*C_Y^{-1} *H*C_U
% where  C_Y = C_V + H*C_U*H'; 
% In code:    Kmat = C_U *H'*C_Y^{-1}; 
%             mupost  = muprior + Kmat * (yn - bn(nodes))
%             covpost = covU - Kmat*H*C_U
% weight of the particles
%        F(x1,mupior,mupost,yobs) = -1/2 * 
%         ( norm(chol_covU\(x1-muprior),2)^2 + norm(yobs-x1(nodes))^2/stdObs^2
%         +1/2 * (  norm( chol_covpost\(x1 - mupost),2)^2  
%                 + norm( yobs- muprior(nodes),2)^2/stdObs^2 )

chol_priorInv = pmcmcPar.chol_priorInv;  
chol_post     = pmcmcPar.chol_post;
stdObs        = obsPar.stdObs;       nodes = obsPar.nodes;  

[Dx,Np]= size(xt0); ind = 1:Np; 
u1det  = forward_ensemble1step(forward1,xt0,ind); % size(u1det)=[Dx, Np]; ind not needed
mupost = u1det + pmcmcPar.Kmat* (obs1step - u1det(nodes,:));   

% %  ============ updated: statePrior
postCov  = chol_post*chol_post.'; 
statePriorMean = obsPar.statePriormean*ones(size(mupost)); 
statePriorCov  = eye(Dx)*obsPar.statePriorstd^2; 
temp    = (statePriorCov+ postCov); 
mupost  = statePriorCov*(temp\mupost) + postCov*(temp\statePriorMean);   
postCov = statePriorCov* (temp\postCov); 
chol_post= chol(postCov);
% %  ===============================

% samples from optimal importance density 
sample = mupost + chol_post* randn(Dx,Np);  
% log weights of the samples
logweights = weightFn_IMP(sample,u1det,mupost,obs1step,chol_post,chol_priorInv,stdObs,nodes);  
const      = max(logweights); % Subtract the maximum value for numerical stability
wt         = exp(logweights-const);
wt         = wt/sum(wt);
end


function logweights = weightFn_IMP(x1,muprior,mupost,obs,chol_post,chol_priorInv,stdObs,nodes)
% logweights for the optimal sampling
% weight of the particles
%    F(x1,mupior,mupost,yobs) =  
%         -1/2 * ( norm(chol_covU\(x1-muprior),2)^2
%                  + norm(yobs-x1(nodes))^2/stdObs^2         ) 
%         +1/2 * (  norm( chol_covpost\(x1 - mupost),2)^2  
%                 + norm( yobs- muprior(nodes),2)^2/stdObs^2 ) 

tempprior   = chol_priorInv*(x1 - muprior);   % size  = [Dx,Np]
temppost    = chol_post\(x1 - mupost);   
tempObs     = (obs- x1(nodes,:))/stdObs;        % size  =[dimObs, Np]  
tempObs1    = (obs- muprior(nodes,:) )/stdObs; 

tempprior   = sum(tempprior.^2,1);  
temppost    = sum(temppost.^2,1); 
tempObs     = sum(tempObs.^2,1);
tempObs1    = sum(tempObs1.^2,1); 

F = - tempprior - tempObs + temppost + tempObs1; 
logweights = F/2; 
end