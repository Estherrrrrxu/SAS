
function [covfreq, ci_cov]= coverageFrequency(samples, quantile,clmf)
% % calculate the covarge frequency
% clmf   = true climate field
% 1. Calculate CI of posterior for given value (e.g. 0.5, 0.9) for each space-time point
% 2. For each space-time point: Set ci_cov to 1 if true value is inside CI or to 0 if not
% 3. Coverage frequency = mean(ci_cov)
ci = calculate_ci(samples, quantile);
num_points = length(ci(:,1));
ci_min = ci(:,1); ci_max = ci(:,2); 
ci_cov = zeros(num_points,1);
for i = 1:num_points
    if (ci_min(i) <= clmf(i) && ci_max(i) >= clmf(i) )
        ci_cov(i) = 1; 
    end
end
covfreq = mean(ci_cov);
return

function ci = calculate_ci(samples, quantile)
% Compute credible interval
[num_points, num_samples]= size(samples);    % Nils: different order here
ci = zeros(num_points,2);  % =[ci_min ci_max]
numqntl = ceil(quantile*num_samples); 
for i=1:num_points
    sample_pti = sort(samples(i,:)); 
    tmp        = [sample_pti(1), sample_pti(numqntl)]; 
    j1 = 1;    
    while (numqntl+ j1 <= num_samples)
         j2 = numqntl+ j1; 
         if ( sample_pti(j2) - sample_pti(j1) ) < (tmp(2) - tmp(1))
             tmp = [sample_pti(j1), sample_pti(j2)];
         end
         j1 = j1+1; 
    end
    ci(i,:) = tmp; 
end
return
