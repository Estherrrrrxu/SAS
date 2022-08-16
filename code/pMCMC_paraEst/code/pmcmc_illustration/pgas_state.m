function [X] = pgas_state(numMCMC, y, N, q0, r0, plotOn)
% Runs the PGAS algorithm,
%
%   F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for
%   Particle Gibbs", Proceedings of the 2012 Conference on Neural
%   Information Processing Systems (NIPS), Lake Taho, USA, 2012.
%
% The function returns the sample paths of (x_{1:T}).

T = length(y);
X = zeros(numMCMC,T);

% Initialize the state by running a PF
[particles, w] = cpf_as(y, q0, r0, N, X(1,:));
% Draw J
J = find(rand(1) < cumsum(w(:,T)),1,'first');
X(1,:) = particles(J,:);


% Run MCMC loop
reverseStr = [];
for(k = 2:numMCMC)
    reverseStr = displayprogress(100*k/numMCMC, reverseStr);
    
    % Run CPF-AS
    [particles, w] = cpf_as(y, q0, r0, N, X(k-1,:));
    % Draw J (extract a particle trajectory)
    J = find(rand(1) < cumsum(w(:,T)),1,'first');   
    X(k,:) = particles(J,:);    
end

end
%--------------------------------------------------------------------------
function [x,w] = cpf_as(y, q, r, N, X)
% Conditional particle filter with ancestor sampling
% Input:
%   y - measurements
%   q - process noise variance
%   r - measurement noise variance
%   N - number of particles
%   X - conditioned particles - if not provided, un unconditional PF is run

conditioning = (nargin > 4);
T = length(y);
x = zeros(N, T); % Particles
a = zeros(N, T); % Ancestor indices
w = zeros(N, T); % Weights
x(:,1) = 0; % Deterministic initial condition
x(N,1) = X(1); % Set the 1st particle according to the conditioning

for(t = 1:T)
    if(t ~= 1)
        ind = resampling(w(:,t-1));
        ind = ind(randperm(N));
        xpred = f(x(:, t-1),t-1);
        x(:,t) = xpred(ind) + sqrt(q)*randn(N,1);
        if(conditioning)
            x(N,t) = X(t); % Set the N:th particle according to the conditioning
            % Ancestor sampling
            m = exp(-1/(2*q)*(X(t)-xpred).^2);
            w_as = w(:,t-1).*m;
            w_as = w_as/sum(w_as);
            ind(N) = find(rand(1) < cumsum(w_as),1,'first');
        end
        % Store the ancestor indices
        a(:,t) = ind;
    end
    % Compute importance weights
    ypred = h(x(:,t));
    logweights = -1/(2*r)*(y(t) - ypred).^2; % (up to an additive constant)
    const = max(logweights); % Subtract the maximum value for numerical stability
    weights = exp(logweights-const);
    w(:,t) = weights/sum(weights); % Save the normalized weights
end

% Generate the trajectories from ancestor indices
ind = a(:,T);
for(t = T-1:-1:1)
    x(:,t) = x(ind,t);
    ind = a(ind,t);
end
end

%-------------------------------------------------------------------
function reverseStr = displayprogress(perc,reverseStr)
msg = sprintf('%3.1f', perc);
fprintf([reverseStr, msg, '%%']);
reverseStr = repmat(sprintf('\b'), 1, length(msg)+1);
end
