function [q,r,X] = pmmh(numMCMC, y, prior, prop, N, qinit, rinit, q0, r0, plotOn)
% Runs the PMMH algorithm,
%
%   C. Andrieu, A. Doucet and R. Holenstein, "Particle Markov chain Monte
%   Carlo methods" Journal of the Royal Statistical Society: Series B,
%   2010, 72, 269-342.
%
% The function returns the sample paths of (q, r, x_{1:T}).

T = length(y);
q = zeros(numMCMC,1);
r = zeros(numMCMC,1);
X = zeros(numMCMC,T);
loglik = zeros(numMCMC,1);
% Initialize the parameters
q(1) = qinit;
r(1) = rinit;
% Initialize the state & likelihood by running a PF
[particles, w, loglik(1)] = pf(y, q(1), r(1), N);
% Draw J
J = find(rand(1) < cumsum(w(:,T)),1,'first');
X(1,:) = particles(J,:);

if(plotOn)
    figure;clf;
    subplot(311);
    plot([1 numMCMC], q0*[1 1],'k-'); hold on;
    xlabel('Iteration');
    ylabel('q');
    title('A sample of the Chain: PMMH')
    subplot(312);
    plot([1 numMCMC],r0*[1 1],'k-'); hold on;
    xlabel('Iteration');
    ylabel('r');    
    subplot(313); hold on;
    xlabel('Iteration');
    ylabel('log-likelihood');
end

% Run MCMC loop
reverseStr = [];
for(k = 2:numMCMC)
    reverseStr = displayprogress(100*k/numMCMC, reverseStr);
    
    % Propose a parameter
    q_prop = q(k-1) + prop.sigma_q*randn(1);
    r_prop = r(k-1) + prop.sigma_r*randn(1);

    if(q_prop <= 0 || r_prop <= 0) % Prior probability 0, reject
        accept = false;
    else        
        % Run a PF to evaluate the likelihood
        [particles, w, loglik_prop] = pf(y, q_prop, r_prop, N);
        
        acceptprob = exp(loglik_prop - loglik(k-1)); % Likelihood contribution
        acceptprob = acceptprob * ... % Prior contribution
            igampdf(q_prop,prior.a,prior.b)*igampdf(r_prop,prior.a,prior.b) /...
            (igampdf(q(k-1),prior.a,prior.b)*igampdf(r(k-1),prior.a,prior.b));
        accept = rand(1) < acceptprob;
    end
            
    if(accept)
        q(k) = q_prop;
        r(k) = r_prop;
        % Draw J (extract a particle trajectory)
        J = find(rand(1) < cumsum(w(:,T)),1,'first');
        X(k,:) = particles(J,:);
        loglik(k) = loglik_prop;
    else
        q(k) = q(k-1);
        r(k) = r(k-1);
        X(k,:) = X(k-1,:);
        loglik(k) = loglik(k-1);        
    end
    
    % Plot
    if(plotOn)
        if(accept)
            style = 'r.';
        else
            style = 'm.';
        end
        subplot(311);        
        plot(k, q(k),style);hold on;
        xlim([0 ceil(k/100)*100]);
        subplot(312);
        plot(k, r(k),style);hold on;
        xlim([0 ceil(k/100)*100]);
        drawnow;
        subplot(313);
        plot(k, loglik(k),style);hold on;
        xlim([0 ceil(k/100)*100]);
        drawnow;          
    end
end
end
%--------------------------------------------------------------------------
function [x,w,loglik] = pf(y, q, r, N)
% Particle filter
% Input:
%   y - measurements
%   q - process noise variance
%   r - measurement noise variance
%   N - number of particles

T = length(y);
x = zeros(N, T); % Particles
a = zeros(N, T); % Ancestor indices
w = zeros(N, T); % Weights
x(:,1) = 0; % Deterministic initial condition
loglik = 0;

for(t = 1:T)
    if(t ~= 1)
        ind   = resampling(w(:,t-1));
        xpred = f(x(:, t-1),t-1);
        x(:,t) = xpred(ind) + sqrt(q)*randn(N,1);        
        a(:,t) = ind; % Store the ancestor indices
    end
    % Compute importance weights
    ypred = h(x(:,t));
    logweights = -1/2*log(2*pi*r) - 1/(2*r)*(y(t) - ypred).^2;
    const = max(logweights); % Subtract the maximum value for numerical stability
    weights = exp(logweights-const);
    % Compute loglikelihood
    loglik = loglik + const + log(sum(weights)) - log(N);        
    w(:,t) = weights/sum(weights); % Save the normalized weights
end

% Generate the trajectories from ancestor indices
ind = a(:,T);
for(t = T-1:-1:1)
    x(:,t) = x(ind,t);
    ind = a(ind,t);
end
end
%--------------------------------------------------------------------------
function y = igampdf(x,a,b)
% IGAMPDF Inverse Gamma probability density funciton
%    Y = IGAMPDF(X,A,B) returns the inverse Gamma probability density
%    function with shape and scale parameters A and B, respectively, at the
%    values in X.
y = exp(a*log(b) - gammaln(a) - (a+1)*log(x) - b./x);
end
%-------------------------------------------------------------------
function reverseStr = displayprogress(perc,reverseStr)
msg = sprintf('%3.1f', perc);
fprintf([reverseStr, msg, '%%']);
reverseStr = repmat(sprintf('\b'), 1, length(msg)+1);
end