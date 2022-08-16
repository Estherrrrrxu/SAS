function [X] = pgas_state(numMCMC, obs, Np, para,Phi_n)
% Runs the PGAS algorithm,
%
%   F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for
%   Particle Gibbs", Proceedings of the 2012 Conference on Neural
%   Information Processing Systems (NIPS), Lake Taho, USA, 2012.
%
% The function returns the sample paths of (x_{1:T}).

T = length(obs);
X = zeros(numMCMC,T);

% Initialize the state by running a PF
[particles, w] = cpf_as(obs, para,Phi_n, Np, X(1,:));
% Draw J
J = find(rand(1) < cumsum(w(:,T)),1,'first');
X(1,:) = particles(J,:);


% Run MCMC loop
reverseStr = [];
for(k = 2:numMCMC)
    reverseStr = displayprogress(100*k/numMCMC, reverseStr);
    
    % Run CPF-AS
    [particles, w] = cpf_as(obs, para,Phi_n, Np, X(k-1,:));
    % Draw J (extract a particle trajectory)
    J = find(rand(1) < cumsum(w(:,T)),1,'first');   
    X(k,:) = particles(J,:);    
end

end
%--------------------------------------------------------------------------
function [x,w] = cpf_as(y, para, Phi_n, Np, X)
% Conditional particle filter with ancestor sampling
% Input:
%   y - measurements
%   q - process noise variance
%   r - measurement noise variance
%   N - number of particles
%   X - conditioned particles - if not provided, un unconditional PF is run
%   Phi_n = @(XX,VV,tn) Terms(XX,VV)*coefs+fnt(tn);    % Phi_n dependes on time


stdS   = para.sigmaV;    %  std of the noise in state model
stdObs = para.sigmaW;    %  std of the observatin noise
p      = para.p;
q      = para.q; 
mLag   = max(p,q);

conditioning = (nargin > 4);
T = length(y);
x = zeros(Np, T); % Particles
a = zeros(Np, T); % Ancestor indices
w = zeros(Np, T); % Weights
x(:,1:mLag) = randn(Np,mLag);   % Deterministic/random IC 
x(Np,:) = X; % Set the 1st particle according to the conditioning
 
rng(12);
v = stdS*randn(Np,T); % a noise sequence

t=1; 
sFlag =0 ; % singular flag
singN =0; % singular increment distribution
while t<=T
    if(t >mLag)
        ind = resampling(w(:,t-1));
        ind = ind(randperm(Np));        
        XX = x(:,t-mLag:t-1); VV = v(:,t-mLag:t-1); 
        xpred = Phi_n(XX,VV,t);       % non-Markov model 
        x(:,t) = xpred(ind) + v(:,t); 
        if(conditioning)
            x(Np,t) = X(t); % Set the N:th particle according to the conditioning
            % Ancestor sampling
            wexp = 1/(2*stdS^2)*(X(t)-xpred).^2
%%% Replace the following by two lines below: use exponent to avoid singularity             
%             m     = exp(wexp-max(wexp) );     % "renormalize" to avoid singular
%             w_as = w(:,t-1).*m;
%             if sum(w_as) ==0         % count tries of resample
%                 singN = singN +1; % disp(singN);
%                 if singN >10^3
%                     disp('Singular increment distribution. Stopped.!!!!!!!');
%                     sFlag =1;
%                     return;
%                 end
%                 v(:,t) = randn(Np,1);
%                 continue;
%             end
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
    ypred = fnObs_nl(x(:,t));
    logweights = -1/(2*stdObs^2)*(y(t) - ypred).^2; % (up to an additive constant)
    const = max(logweights); % Subtract the maximum value for numerical stability
    weights = exp(logweights-const);
    w(:,t) = weights/sum(weights); % Save the normalized weights
    t = t+1;
end

% Generate the trajectories from ancestor indices
ind = a(:,T);
for(t = T-1:-1:1+mLag)
    x(:,t) = x(ind,t);
    ind = a(ind,t);
end
x(:,1:mLag) = x(ind,1:mLag); 

end

%-------------------------------------------------------------------
function reverseStr = displayprogress(perc,reverseStr)
msg = sprintf('%3.1f', perc);
fprintf([reverseStr, msg, '%%']);
reverseStr = repmat(sprintf('\b'), 1, length(msg)+1);
end
