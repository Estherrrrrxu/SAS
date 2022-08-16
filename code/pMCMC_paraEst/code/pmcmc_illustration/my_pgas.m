function [q,r,X] = my_pgas(numMCMC, y, prior, N, qinit, rinit, q0, r0, plotOn)
% Runs the PGAS algorithm, 
%
%   F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for
%   Particle Gibbs", Proceedings of the 2012 Conference on Neural
%   Information Processing Systems (NIPS), Lake Taho, USA, 2012.
%
% The function returns the sample paths of (q, r, x_{1:T}).
% 
% Changes: resample the trajectory at each step

T = length(y);
q = zeros(numMCMC,1);
r = zeros(numMCMC,1);
X = zeros(numMCMC,T);
% Initialize the parameters
q(1) = qinit;
r(1) = rinit;
% Initialize the state by running a PF
[particles, w] = cpf_as(y, q(1), r(1), N, X(1,:));
% Draw J
J = find(rand(1) < cumsum(w(:,T)),1,'first');
X(1,:) = particles(J,:);

if(plotOn)
    figure(1);clf(1);
    subplot(211);
    plot([1 numMCMC], q0*[1 1],'k-'); hold on;
    xlabel('Iteration');
    ylabel('q');
    title('A sample of the Chain: PGAS')
    subplot(212);
    plot([1 numMCMC],r0*[1 1],'k-'); hold on;
    xlabel('Iteration');
    ylabel('r');    
end

% Run MCMC loop
reverseStr = [];
for(k = 2:numMCMC)
    reverseStr = displayprogress(100*k/numMCMC, reverseStr);
    
    % Sample the parameters (inverse gamma posteriors)
    err_q = X(k-1,2:T) - f(X(k-1,1:T-1), 1:(T-1));
    q(k) = igamrnd(prior.a + (T-1)/2, prior.b + err_q*err_q'/2);
    err_r = y - h(X(k-1,:));
    r(k) = igamrnd(prior.a + T/2, prior.b + err_r*err_r'/2);
    % Run CPF-AS
    [particles, w] = cpf_as(y, q(k), r(k), N, X(k-1,:));
    % Draw J (extract a particle trajectory)
    J = find(rand(1) < cumsum(w(:,T)),1,'first');   
    X(k,:) = particles(J,:);    
    % Plot
    if(plotOn)
        subplot(211);
        plot(k, q(k),'r.');hold on;
        xlim([0 ceil(k/100)*100]);
        subplot(212);
        plot(k, r(k),'r.');hold on;
        xlim([0 ceil(k/100)*100]);
        drawnow;       
    end
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
        ind = resampling(w(:,t-1));  % resample every step
        ind = ind(randperm(N));
        x   = x()
        xpred = f(x(:, t-1),t-1);
        x(:,t) = xpred(ind) + sqrt(q)*randn(N,1); %the past traj un-resampled
                   % and this is done at the end. 
                   % why not do it here? - ancestor sampling 
                   % the low weighted particle is dropped in xpred(ind) 
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


end


%-------------------------------------------------------------------
function R = igamrnd(a,b,varargin)
% IGAMRND Random arrays from inverse gamma distribution.
%    R = IGAMRND(A,B) returns an array of random numbers chosen from the
%    inverse gamma distribution with shape parameter A and scale parameter B.
% 
%    R = GAMRND(A,B,M,N,...) or R = GAMRND(A,B,[M,N,...]) returns an
%    M-by-N-by-... array.

R = 1./gamrnd(a,1/b,varargin{:});
end
%-------------------------------------------------------------------
function reverseStr = displayprogress(perc,reverseStr)
msg = sprintf('%3.1f', perc);
fprintf([reverseStr, msg, '%%']);
reverseStr = repmat(sprintf('\b'), 1, length(msg)+1);
end
