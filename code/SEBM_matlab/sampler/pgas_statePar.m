function [X,ess,theta] = pgas_statePar(numMCMC,obs,t0t1,regions,femPar,Np,dt,...
                         tN,theta0,varObs, prior,sp)
% Runs the PGAS algorithm to estimate states and parameters
%  % numMCMC     - number steps in MCMC
%  % obs         - observations:   size =[ttN,rN]
%  % t0t1,regions- cooresponding to obs
%  % femPar      - all FEM related 
%  % Np          - number of particles
%  % varObs      - variance of observation noise
% Output
%  %  X     - samples of paths of state variable
%  %  ess   - effectitve sample size of the PF traj in each MCMC step
%  %  theta - samples of theta

% Reference
%   F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for
%   Particle Gibbs", Proceedings of the 2012 Conference on Neural
%   Information Processing Systems (NIPS), Lake Taho, USA, 2012.
%
% Last updated by Fei Lu, 2018-6-12

Prec_chol = femPar.Prec_chol;  % Cholesky of the precision matrix
Dx  = length(Prec_chol(1,:));
X   = zeros(Dx,tN,numMCMC);

K   = length(theta0); 
theta = zeros(K,numMCMC);

ttN = length(t0t1(:,1));   % number of SMC time steps = obs Times
ess = zeros(1,numMCMC);
femPar.bounds= prior.bounds; % lower and upper bounds of state

% Initialize the state by running a PF
[particles,w,~] = cpf_as(obs,t0t1,regions,femPar,dt,Np,varObs,theta0, X(:,:,1));
ess(1)    = 1/(w(:,end)'*w(:,end));  % effective sample size
% Draw J
J = find(rand(1) <= cumsum(w(:,ttN)),1,'first');
X(:,:,1) = particles(:,:,J);  
if sp ==0 % estimate parameter+ state; Otherwise, estimate state with TruePar  
    theta(:,1)    = sampleTheta_Bayes(X(:,:,1),femPar,K,dt,prior); 
    theta0 = theta(:,1)';  
end

% Run MCMC loop
reverseStr = [];
for k = 2:numMCMC
      reverseStr = displayprogress(100*k/numMCMC, reverseStr);
    % Run CPF-AS
    [particles,w,~] = cpf_as(obs,t0t1,regions,femPar,dt,Np,varObs,theta0,X(:,:,k-1));
    ess(k)    = 1/(w(:,end)'*w(:,end));  % effective sample size
    % Draw J (extract a particle trajectory)
    J = find(rand(1) < cumsum(w(:,ttN)),1,'first');
    X(:,:,k) = particles(:,:,J);  
    
    if sp ==0  % estimate parameter+ state; Otherwise, estimate state with TruePar
        theta0    = sampleTheta_Bayes(X(:,:,k),femPar,K,dt, prior);
        theta(:,k)= theta0;
        theta0    = theta0';
    end
end

end


%--------------------------------------------------------------------------
function [x,w,sFlag] = cpf_as(obs,t0t1,regions,femPar,dt,Np,varObs,theta,X1)
% Conditional particle filter with ancestor sampling
% Input:
%   obs  - measurements: size = [# of time intervals, # regions]
%                            = [length(t0t1(:,1)), length(regions(1,:))]
%   t0t1 - start and end of time intervals  ttNx2; connected (for SMC)
%          regions- nodes in regions, size= [ttN*rN, eN]
%   varObs- measurement noise variance
%   Np    - number of particles
%   X1     - conditioned particles - if not provided, un unconditional PF is run
chol_PAB  = femPar.Prec_chol*femPar.ABmat; 
elements3 = femPar.elements3; 
TkVol     = femPar.tri_areas/3; 
forward   = femPar.forward; 
forward1  = @(U0,dt,tn) forward(U0,dt,tn,theta); 
stdF      = femPar.stdF; stdF2 = stdF.^2; 


conditioning = 1; % (nargin > 9);
[Dx,T,~]     = size(X1); 
X_Np         = reshape(X1,[Dx,T]);  % size(X1)=[ Dx,T,1]->>[Dx T]. Ref path
x  = zeros(Dx,T,Np); % Particles

% % ============= NEW!==========
x(:,1,:) = 1+0*randn(Dx,Np);    % positive initial condition

x(:,:,Np) = X1(:,:,1); % Set the Np:th particle according to the conditioning

ttN = length(t0t1(:,1));   % number of SMC time steps = obs Times 
rN  = length(regions(:,1)) / ttN; 

a  = zeros(Np,ttN);   % Ancestor indices
w  = zeros(Np, ttN);  % Weights

tt = 1;      % observation time index 
sFlag = 0 ;  % singular indicator
singN    = 0; % singular increment distribution


% % ============= NEW!==========
% NOTE on t0t1: here we assume t0t1, an Lx2 array, are start and end times 
% of time intervals, and Importantly, each start = the previous end, i.e. 
%            t0t1(l,1) = t0t1(l-1,2) 
while tt <= ttN
    t0= t0t1(tt,1); t1 = t0t1(tt,2);  steps =t1 - t0; 
    if tt==1   %%% initialization up to the first t1 
        xt0 = reshape(x(:,t0,:), Dx,Np); ind = 1:Np; 
        [xpred,~] = forward_ensemble(forward1,xt0,dt,steps,ind); % size(xpred)=[Dx,steps+1, Np]
        x(:,t0+1:t1,:) = xpred(:,2:end,:); % NEW: removed t0 overlap
    else % % tt>1 
        ind = myResampling(w(:,tt-1));
        ind = ind(randperm(Np));        
        xt0 = reshape(x(:,t0,:), Dx,Np);  % resampled particles as inital condition
        [xpred,u1det] = forward_ensemble(forward1,xt0,dt,steps,ind); % -Checked: 6/19-
                     % size(u1det) = [Dx,Np]
                     % u1det is the 1-step deterministic forward of the particles
        x(:,t0+1:t1,:) = xpred(:,2:end,:);   % NEW: removed t0 overlap 
        if(conditioning)
            x(:,t0+1:t1,Np) = X_Np(:,t0+1:t1); % Set the Np-th particle as conditioning
            % Ancestor sampling
            X_Np1= X_Np(:,t0+1);    % NEW: only one-step probability needed
            wexp = multiPriorWt(X_Np1,u1det,chol_PAB,dt,stdF2); 
%%% Replace the following by two lines below: use exponent to avoid singularity
%            wexp = wexp - max(wexp);   %% --DONE: use log w(:,t-1) instead ====
%            m    = exp( wexp/2);
%            w_as = w(:,tt-1).*m;     % m and w(:,t-1) can be singular!!!!
%            if sum(w_as) ==0         % count tries of resample
%                singN = singN +1;    % disp(singN);
%                if singN >10^3
%                    disp('Singular increment distribution. Stopped.'); 
%                    sFlag =1;       return;    
%                end
%               continue; 
%            end 
            wexp = wexp + logweights'; % use the exponents to avoid singular
            w_as = exp(wexp- max(wexp));  
            
            w_as = w_as/sum(w_as);
            temp = find(rand(1) <= cumsum(w_as),1,'first');
             if isempty(temp); keyboard; end
            ind(Np) = temp;
        end
        % Store the ancestor indices --- starting from tt=2
        a(:,tt) = ind;
    end
    figure(17); clf; titl='Before traj-resample'; plotTrajpf(x,5,titl); pause(0.05)
    % Compute importance weights:
    UU     = x(:,t0:t1,:); % Let us Right-point approximation for now
    %NILS: Technically need to use resampled version of x(:,t0,:), i.e. x(:,t0,ind) and then x(:,(t0+1):t1,:); as long as we use right-point approximation, this is not necessary, as in this case x(:,t0,:) is not used in the observation operator
    region1= regions(rN*(tt-1)+(1:rN),:);   
    ypred  = fnObs_ensemble(UU, region1, elements3, TkVol); % size=[rN,Np]
    logweights = -1/(2*varObs)*sum( (obs(tt,:)' - ypred).^2,1); 
    const = max(logweights); % Subtract the maximum value for numerical stability
    weights = exp(logweights-const);
    w(:,tt) = weights/sum(weights); % Save the normalized weights
    tt = tt+1; 
end

% Generate the trajectories from ancestor indices
ind = a(:,ttN);
for tt = ttN-1:-1:1
    t0= t0t1(tt,1); t1 = t0t1(tt,2);
    if tt>1
        x(:,t0+1:t1,:) = x(:,t0+1:t1,ind);
    else
        x(:,t0:t1,:) = x(:,t0:t1,ind);
    end
    ind = a(ind,tt);
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

%-------------------------------------------------------------------
function reverseStr = displayprogress(perc,reverseStr)
msg = sprintf('%3.1f', perc);
fprintf([reverseStr, msg, '%%']);
reverseStr = repmat(sprintf('\b'), 1, length(msg)+1);
end


