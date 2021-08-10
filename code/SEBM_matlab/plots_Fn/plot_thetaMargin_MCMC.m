function plot_thetaMargin_MCMC(sampleFilename,datapath1)
% plot the marginals of theta as the length of the Markov chain increases

% mkSz =12; ftsz = 14; 
tInd = [5,1]*1000; 

load(sampleFilename);

fig_theta = strcat(datapath1,'fig_MCtheta');

sampleALL = theta(:,300:numMCMC); 
titl    = {'Marginal posterior \theta_0','Marginal posterior \theta_1',...
          'Marginal posterior \theta_4',};   
K = 3 ; 
figure; % set(gcf, 'Position',  [100, 1000, 400, 300]);
linS = {'r:','b--'}; % linestyle in for loop
linW = [2,1];        
for kk =1:K
    subplot(K,1,kk); 
    samples = sampleALL(kk,:);  
    minS    = min(samples); maxS  = max(samples); gap =(maxS-minS)/100; 
    points  = minS-gap:gap:maxS+gap; 
    [f,xi]  = ksdensity(samples, points);   
    plot(xi,f,'k','linewidth',1); hold on; 
    for i = 1:length(tInd)
        samples = sampleALL(kk,1:tInd(i));
        [f,xi]  = ksdensity(samples, points);  
        plot(xi,f,linS{i},'linewidth',linW(i)); hold on;
    end  
    xlb = prior.lb(kk); xub = prior.ub(kk); cc= xub-xlb;  
    xlim([xlb-cc/5 xub+cc/5]);  
    title(titl{kk}); 
  
    if kk==1
        legend({'L = 10k','L = 5k','L = 1k'},'Location','northeast'); % legend('boxoff');
    end
    setPlot_fontSize;
end

print(fig_theta,'-depsc');

return
