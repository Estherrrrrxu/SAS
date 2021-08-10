function [X,ess,theta] = pgas_stateParNodes_statePrior(pmcmcPar,obs,obsPar,femPar,prior,progressON)
% Runs the PGAS algorithm to estimate states and parameters
%  % numMCMC     - number steps in MCMC
%  % obs         - observations:   size =[rN,ttN]  -- Diff from region obs!!
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
% Last updated by Fei Lu, 2019-1-12 % 2018-12-18
sp      = pmcmcPar.sp;       % 0 estimate the parameter; 1, estimate states with true para
numMCMC = pmcmcPar.numMCMC;  
tN      = obsPar.tN+1;      

dt     = femPar.dt; 
Prec_chol = femPar.Prec_chol;  % Cholesky of the precision matrix
Dx     = length(Prec_chol(1,:)); 
X      = zeros(Dx,tN,numMCMC);    % samples of state
theta0 = prior.initialguess;    K      = length(theta0);  % in-PDE nl_fn, 1xK
theta  = zeros(K,numMCMC);     % samples of parameter

ttN = length(obs(1,:)) ;   % number of SMC time steps = obs Times
ess = zeros(1,numMCMC);
femPar.bounds= prior.statebounds; % lower and upper bounds of state

% Initialize the state by running a PF
[particles,w] =  pf_NodesIPF_statePrior(obs,obsPar,femPar,pmcmcPar,theta0);
ess(1)        = 1/(w(:,end)'*w(:,end));  % effective sample size
% Draw J
J = find(rand(1) <= cumsum(w(:,end)),1,'first');
X(:,:,1) = particles(:,:,J);  
if sp ==0 % estimate parameter+ state; Otherwise, estimate state with TruePar  
    theta(:,1) = sampleTheta_Bayes(X(:,:,1),femPar,K,dt,prior,progressON); 
    theta0     = theta(:,1)';  
end
% figure(18);clf; titl='After traj-resample: AS'; plotTrajpf(particles,5,titl);pause(0.05)
% Run MCMC loop
reverseStr = []; 
for k = 2:numMCMC
    if ~exist('progressON','var') || progressON==1 
      reverseStr = displayprogress(100*k/numMCMC, reverseStr); 
    end
    % if k==32; keyboard; end
    % Run CPF-AS
    if strcmp(pmcmcPar.sampler, 'IPF')
        [particles,w,~] = cpf_asNodes_IPF_statePrior(obs,obsPar,femPar,pmcmcPar,theta0,X(:,:,1));
    else
        [particles,w,~] = cpf_asNodes_SIR(obs,obsPar,femPar,pmcmcPar,theta0,X(:,:,1));
    end
    wEnd = w(:,end);  if sum(wEnd) == 0; fprintf('0-weight!\n'); keyboard; end
    ess(k)    = 1/(wEnd'*wEnd);  % effective sample size
    % Draw J (extract a reference trajectory )
    J = find(rand(1) <= cumsum(wEnd),1,'first');
    X(:,:,k) = particles(:,:,J);  
    
    if sp ==0  % estimate parameter+ state; Otherwise, estimate state with TruePar
        theta0    = sampleTheta_Bayes(X(:,:,k),femPar,K,dt, prior,progressON);
        theta(:,k)= theta0;        theta0    = theta0';
    end
end
end

%--------------------------------------------------------------------------


%-------------------------------------------------------------------
function reverseStr = displayprogress(perc,reverseStr)
msg = sprintf('%3.1f', perc);
fprintf([reverseStr, msg, '%%']);
reverseStr = repmat(sprintf('\b'), 1, length(msg)+1);
end


