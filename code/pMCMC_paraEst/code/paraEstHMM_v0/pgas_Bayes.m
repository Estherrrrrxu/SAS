function [X,theta,svar,ess] = pgas_Bayes(numMCMC, obs, Np, ssmPar)
% Runs the PGAS algorithm, SIR in the SMC
%
%   F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for
%   Particle Gibbs", Proceedings of the 2012 Conference on Neural
%   Information Processing Systems (NIPS), Lake Taho, USA, 2012.
%
% The function returns the sample paths of (x_{1:T}).
% Output
%     X  -- the Markov chains of (x_{1:T}). size = [numMCMC,T]

T = length(obs);         % T=para.N; 
X = zeros(numMCMC,T);
theta = zeros(numMCMC,length(ssmPar.thetaTrue)); 
svar  = zeros(numMCMC,1);                         
ess   = zeros(numMCMC,T);

Ltheta = length(ssmPar.thetaTrue);
thetaInd  = ssmPar.thetaInd;      % the index of parameters to be estimated
theta0    = ssmPar.thetaTrue;     
theta0(thetaInd) = ssmPar.theta0(thetaInd); 

sampleVar = 1;             %  if 1, estimate variance of state model  
varV      = 0.1*sampleVar+ssmPar.sigmaV^2; % variance of the state model noise 

% Initialize the state by running a PF
[Xpaths,Vpaths,Termspaths,ft,w,ess1] = cpf_as(obs, ssmPar,Np,theta0,varV, X(1,:));
% Draw J from the weights. Traj J will be reference traj in next step SMC
J  = find(rand(1) < cumsum(w(:,T)),1,'first');
X1 = Xpaths(J,:);           V1 = Vpaths(J,:); 
T1 = reshape(Termspaths(J,:,:),Ltheta,[]);  % paths of terms, for sampling paramters
X(1,:) = X1;       ess(1,:) = ess1; 
[thetaEst,varV]  = samplePar_Bayes(X1,V1,T1,ft,ssmPar,sampleVar); 
theta0(thetaInd) = thetaEst;  
theta(1,:)       = theta0;  svar(1) = varV; 

% Run MCMC loop
reverseStr = [];
for k = 2:numMCMC 
    reverseStr = displayprogress(100*k/numMCMC, reverseStr);
    % % Run CPF-AS
    [Xpaths,Vpaths,Termspaths,ft,w,ess1] = cpf_as(obs,ssmPar,Np,theta0,varV, X(k-1,:));
    % % Draw J (extract a particle trajectory)
    J  = find(rand(1) < cumsum(w(:,T)),1,'first'); 
    X1 = Xpaths(J,:); V1 = Vpaths(J,:);
    T1 = reshape(Termspaths(J,:,:),Ltheta,[]);  
    X(k,:) = X1;  ess(k,:) = ess1; 
    [thetaEst,varV] = samplePar_Bayes(X1,V1,T1,ft,ssmPar,sampleVar);
    theta0(thetaInd) = thetaEst; 
    theta(k,:) = theta0;  svar(k) = varV; 
end

end

%-------------------------------------------------------------------
function reverseStr = displayprogress(perc,reverseStr)
msg = sprintf('%3.1f', perc);
fprintf([reverseStr, msg, '%%']);
reverseStr = repmat(sprintf('\b'), 1, length(msg)+1);
end
