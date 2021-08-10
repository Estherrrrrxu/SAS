function thetaTrue = sampleThetaTrue(prior)
% sample true values for theta from prior

% thetaTrue = [0.1,-1,0.4,-1,0.2]; % original test
if prior.flag ==1
    temp      = prior.mu';    % should sample from prior later  
    thetaTrue = [exp(temp(1)), temp(2:4), -exp(temp(5))]; % 1x5 vector
elseif prior.flag ==0     % Gaussian
    % temp  = prior.mu + randn(size(prior.mu)).*prior.std; thetaTrue = temp';  
    thetaTrue = theta3std(prior); 
    
elseif prior.flag ==2     % uniform
    temp = prior.lb +rand(size(prior.mu)).*(prior.ub-prior.lb); thetaTrue = temp'; 
end

end




function theta = theta3std(prior)
% generate theta with threshold. For test
K = length(prior.mu); 
maxIter=100; m=1;
while  m<=maxIter
    temp = randn(K,1);
    indicator   = sum(abs(temp)<[3;3;3]);
    if  indicator == K;    break;     end
    fprintf('Theta out of range! resampled.\n');
    m = m+1 ;
end
theta  = prior.mu + temp.*prior.std;
if m==maxIter
    fprintf('Theta out of range! Max iteration! Likelihood cov bad\n');
    theta = prior.mu;
end
theta = theta'; 
   
end
