function [x,w,sFlag] = cpf_asNodes_IPF_statePrior(obs,obsPar,femPar,pmcmcPar,theta0,X1)
% Conditional particle filter with ancestor sampling, using Implicit
% particle fiter (equivalent to optimal importance sampling here)
% Input:
%   obs    - measurements: size = [# of time intervals, # regions]
%                            = [length(t0t1(:,1)), length(regions(1,:))]
%   obsPar, femPar, pmcmcpar - parameters for obs, fem and pmcmc
%   theta0  -  value of theta 
%   X1      - conditioned particles - must be provided
%   % previous:  t0t1 - start and end of time intervals  ttNx2; connected (for SMC)
%            regions- nodes in regions, size= [ttN*rN, eN]
% Last updated: 2019-3-5
%
% % State model in Gaussian transition format:
%             U_n = bn + sqrt(dt)*stdF*ABmat\Prec_chol\N(0,Id)
%                 = bn + N(0,C_U),    
%  where   bn  = udet from forward_ensemble.m
%          C_U = dt*stdF^2*Cmat*Cmat^T, with Cmat^{-1} = ABmat*Prec_chol
% % Observation model
%             Y_n = H U_n + V_n

obsPar = statePrior(obsPar,obs); % obtain state prior (mean&std) from obs

pgas = pmcmcPar.pgas;     %  1 for ancester sampling; 0 for cpf without AS

forward1step   = femPar.forward1step;        dt    = femPar.dt; 
forward1  = @(U0) forward1step(U0,theta0); 
chol_PAB  = femPar.Prec_chol*femPar.ABmat; 
stdF      = femPar.stdF;           stdF2 = stdF.^2; 

% initialize state sample (particle) traj's 
Dx        = obsPar.stateDim;  Np  = pmcmcPar.Np;   
T         = obsPar.tN +1;   % length of state traj; obs(tt) is observation of x(tt+1) 
x         = zeros(Dx,T,Np);         % Particles
%  initial ensemble of samples: use obs
nodes     = obsPar.nodes;    nodesrest = setdiff(1:Dx,nodes); 
x(nodes,1,:)     = obs(:,1) + 0.01*randn(length(nodes),Np);
x(nodesrest,1,:)  = 1 +  0.01*randn(length(nodesrest),Np);     

X_Np      = reshape(X1,[Dx,T]);  % size(X1)=[ Dx,T,1]->>[Dx T]. Ref path
x(:,:,Np) = X1(:,:,1);    % Set the Np:th particle as the reference traj
w   = zeros(Np, T);  w(:,1) = ones(Np,1)/Np; % particle Weights; same length as x  
ttN = length(obs(1,:));   % number of SMC time steps = obs Times 
a   = zeros(Np, ttN);     % Ancestor indices

sFlag = 0 ;       % indicator of singular ancestor weights 
for tt =1:ttN     % % ======  NOTE: obs(tt) is observation of x(tt+1) 
    t0 = tt; t1 = t0 +1;  % Observe every step here
    % resamling first: 
    %  - pgas=1 sample ancestor of ref-traj; resample traj by trace back later
    %  - pgas=0: resample traj now
    ind = myResampling(w(:,tt));     ind = ind(randperm(Np));
    indpart = ind(1:Np-1);        % NEW: resample Np-1 particles 
    if tt ==1; ind = 1:Np; end  
    
    % Ancestor sampling for the Np-th particle (reference traj)
    if tt>1 && pgas 
        wexp = multiPriorWt(xt1ref,u1det,chol_PAB,dt,stdF2);        
        wexp = wexp + logweights'; % use the exponents to avoid singular
        w_as = exp(wexp- max(wexp));           w_as = w_as/sum(w_as);
        temp = find(rand(1) <= cumsum(w_as),1,'first');
        if isempty(temp); sFlag=1; fprintf('Singular ancestor weights at t=%i \n',tt);
            disp([wexp,w_as]); keyboard;   temp= Np; 
        end   % set ancester of the Np-th paritcle as itself (should not happen!)
        ind(Np) = temp;
        a(:,tt) = ind;   % ancestor indices -- starting from tt=2
    else 
        x(:,1:t0,1:Np-1) =  x(:,1:t0,indpart);   % pgas=0: resample traj now
% figure(18);clf; titl='After traj-resample: AS'; plotTrajpf(x,5,titl) ;pause(0.05)
    end
    
    xt0    = reshape(x(:,t0,:), Dx,Np);  
    xt1ref = X_Np(:,t0+1);    
    [sample,u1det, wt,logweights] = Optimal_ImpS(forward1,xt0,xt1ref,indpart,obs(:,tt),pmcmcPar,obsPar); 
    x(:,t1,1:Np-1) = sample(:,1:Np-1);  %  Np-th particle not chnaged here 
    % if mod(tt,10)==9; keyboard; end
    w(:,tt+1)      = wt; 
% figure(17); clf; titl='Before traj-resample'; plotTrajpf(x,5,titl); pause(0.05)
end
% PGAS: Generate the trajectories from ancestor indices: 
if pgas;       ind = a(:,ttN);   
    for tt = ttN-1:-1:1 
        t0= tt; t1 = t0+1; 
        if tt>1;   x(:,t0+1:t1,:) = x(:,t0+1:t1,ind);  % avoid overlap
        else;      x(:,t0:t1,:)   = x(:,t0:t1,ind);            end 
        ind = a(ind,tt);
    end
% figure(18);clf; titl='After traj-resample: AS'; plotTrajpf(x,5,titl) ;pause(0.05)
end

end

    
    function [sample,u1det,wt,logweights] = Optimal_ImpS(forward1,xt0,xt1ref,ind,obs1step,pmcmcPar,obsPar)
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

sample = 0*xt0;   sample(:,end) = xt1ref; % set the last as the reference traj  *****
[Dx,Np]= size(xt0);
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
sample(:,1:Np-1) = mupost(:,1:Np-1) + chol_post* randn(Dx,Np-1);  
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