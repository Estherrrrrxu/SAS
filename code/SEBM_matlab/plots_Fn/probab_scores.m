function [crps,es,crps_all,es_all] = probab_scores(samples,trueV,figname1,datapath1,plotON)
% plot the CRPS, ES,CE of the sample
% 
% reference: 
%   1. Pekins and Hakim 2017: Reconstructing paleoclimate fields using online data assimilation with a linear inverse model
%   2. Gneiting and Raftery JASA07: Strictly proper scoring rules, prediction, and estimation

%% CRPS  (continuously ranked probability score for scalar rv)
% For a random variable: CRPS = - 0.5*\E[|X-X'|] +  \E[X-x] (This is CRPS* in GR07)
%  + Using iid sample to approximate:  
%     CRPS  = -1/(2 *M^2)*\sum_{i=1:M, j=1:M} |X^i - X^j|     
%             + 1/M*\sum_{i=1:M} |X^i - x|
%  + If the samples are from a Markov chain, use i = 1:M/2; j=Lag+(1:M/2)
%  + For a whole trajectory:   CRPS = \sum_t CRPS(t)

%% Energy score (crps for vectors)
%     ES  = 1/2* \E[\|Xvec-Xvec'\|^\beta] - \E[\|X-x\|^\beta]
% 

%% CE (coefficient of efficiency) 
%      CE =  1- \E[|x_t - X_t|^2] / \E[x_t - x_mean] 
%  + compute by time averaging over the ergodic trajectory

%% Correlation 
%      Corr = \E[(X_1-X_mean)(v_1 - v_mean)]/ (sigma_x *sigma_v)

beta = 2; 
fig_name = strcat(datapath1,'crps',  figname1);

sz    = size(samples); % [K,tN,M]=size(samples); 
if length(sz) ==3     % [K,tN,M]=size(samples);
    K = sz(1); tN = sz(2); M= sz(3);
    crps_all = zeros(K,tN); es_all = zeros(1,tN); 
    for tt = 1:tN
        sampleKvec = reshape(samples(:,tt,:), [K,M]); 
        for kk=1:K 
            crps_all(kk,tt) = crps_MC(sampleKvec(kk,:), trueV(kk,tt));
        end
        es_all(tt) = EngyScore(sampleKvec,trueV(:,tt),beta); 
    end
    
    if exist('plotON', 'var') && plotON==1
        figure; set(gcf, 'Position',  [100, 1000, 400, 300]);
        plot(1:tN, mean(crps_all),'b-', 1:tN,es_all,'k-.');
        xlabel('Time t'); legend('CRPS','Energy Score'); axis([1,tN,0,0.06])
        print(fig_name,'-depsc');
    end
elseif length(sz) == 2 && min(sz) >2   % [K,M]=size(samples);
    K = sz(1); M = sz(2);
    crps_all = zeros(K,1);  
    for kk=1:K
        crps_all(kk) = crps_MC(samples(kk,:), trueV(kk));
    end
    es_all    = EngyScore(samples,trueV,beta); 
elseif length(sz) == 2 && min(sz) ==1   % [1,M]=size(samples);
    crps_all = crps_MC(samples, trueV);
    es_all    = EngyScore(sample,trueV,beta);
end
    crps = mean( mean(crps_all));  
    es   = mean( mean(es_all));  
end


function [crps,crps0 ] = crps_MC(sample,trueV)
% compute the crps of sample vs trueV, all must be scalor
% Here we assume that sample is from a Markov chain, therefore we use a Lag
%  + Using iid sample to approximate:  
%     CRPS  = -1/(2 *M^2)*\sum_{i=1:M, j=1:M} |X^i - X^j|     
%             + 1/M*\sum_{i=1:M} |X^i - x|
%  + If the samples are from a Markov chain, use i = 1:M/2; j=Lag+(1:M/2)
%     CRPS  = -1/(2 *M/2)*\sum_{i=1:M/2} |X^i - X^(i+M/2)|     
%             + 1/M*\sum_{i=1:M} |X^i - x|
%
% OUTPUT
%    crps   - computed using lag for sample from Markov chain
%    crps0  - computed as if sample is from iid 
M = length(sample);   L= floor(M/2);
ind1  = 1:L-100; ind2 = L+101:2*L;
crps  = -1/2* sum( abs(sample(ind1) - sample(ind2)) )/L + sum(abs(sample-trueV))/L; 

crps0 = 0;  
% for mm=1:M  % good for iid samples
%     temp  = sample - sample(mm); 
%     crps0 = crps0+ sum(abs(temp)); 
% end
% crps0  = crps0/(M^2) - sum(abs(sample-trueV) )/M;   

end

function [ES,es0] = EngyScore(sample,trueV,betap)
% The energy score of sample vector
[~,M] = size(sample);   L= floor(M/2);
ind1  = 1:L; ind2 = L+1:2*L;

temp  = sample(:,ind1) - sample(:,ind2); temp2= sample(:,ind1) - trueV; 
ES    = - 1/2 * sum ( sum(temp.^2).^(betap/2) )/L + sum( sum(temp2.^2).^(betap/2))/L; 

es0 = 0;
% for mm=1:M
%     temp  = sample - sample(:,mm); 
%     es0 = es0+ sum( sum(abs(temp).^2) )^(betap/2) /M; 
% end
% es0  = es0/(2*M) - sum( sum(abs(sample-trueV).^2) )^(betap/2) /M;   

end



