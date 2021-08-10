function [covfreq,ci_cov,update_rates] = coverageFreq_updateRate(filename,quantile,figname1,path)
% calculate the coverage frequency
% clmf   = true climate field
% 1. Calculate CI of posterior for given value (e.g. 0.5, 0.9) for each space-time point
% 2. For each space-time point: Set ci_cov to 1 if true value is inside CI or to 0 if not
% 3. Coverage frequency = mean(ci_cov)
% 4. Compute the acceptance rate
% Nils Weitzel, Fei Lu 2018-5-10

load(filename);  % load Sampledata_tN100.mat 
%% calculate coverage frequency of state estimation
% quantile = 0.9;  
samples = Usample(:,:,burnin:numMCMC); 
samples = reshape( samples,[],length(samples(1,1,:)) ) ;
clmf    = reshape(Utrue,[],1); 

[covfreq,ci_cov] = coverageFrequency(samples, quantile,clmf);
fprintf('state: Quantile, Coverage frequency = (%1.2f,%1.3f) \n', quantile,covfreq);

%% calculate coverage frequency of theta: not so useful. Good for highD state
samples = theta(:,burnin:numMCMC); 
covfreqTheta = coverageFrequency(samples, quantile,thetaTrue); 
fprintf('Theta: Quantile, Coverage frequency = (%1.2f,%1.3f) \n', quantile,covfreqTheta);


%% calculate the update rate 
% - Usample: t=1 is random initial; observation time t=2:tN
% - therefore, we should present update_rate for t=2:tN   
tN = length(Usample(1,:,1)) ; 
update_rates = zeros(1,tN);   % 
 for j= 1:tN 
     diff = Usample(:,j,2:end)-Usample(:,j,1:end-1); 
    update_rates(j) = sum(sum(sqrt(diff.^2) )>0.0001)/(numMCMC-1); 
 end
figure
plot(1:tN-1,update_rates(2:tN),'linewidth',1); 
xlabel('Time'); ylabel('Update rates'); 
title('Update rate of states in MCMC');
fig_rate = strcat('fig_rate',figname1);
fig = gcf;  set(fig,'Position',[100, 1000, 500, 300]); fgca = gca; fgca.FontSize =12; 
print([path,fig_rate],'-depsc');
return


