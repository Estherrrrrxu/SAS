function MAP = plotdensityPrior(sampleArray, trueV,titl,prior)
% plot the marginal densities given by samples

K = length(trueV); MAP = zeros(K,1); 
for kk = 1:K
  subplot(K,1,kk); 
  samples = sampleArray(kk,:); 
  smean   = mean(samples); % std1 = std(samples); 
  minS    = min(samples); maxS  = max(samples); gap =(maxS-minS)/100; 
  points  = minS-gap:gap:maxS+gap; 
  [f,xi]  = ksdensity(samples, points);  
  [~,indMax] = max(f); MAP(kk)=xi(indMax); 
  plot(xi,f); hold on;   
  tmp=get(gca,'ylim'); tmp = [0,tmp(2)];
  plot([trueV(kk) trueV(kk)],tmp*1.2,'k-*');
  plot([smean smean],tmp*1.2,'k-d'); 
  if exist('prior','var')
      if prior.flag==0
          plot([prior.mu(kk) prior.mu(kk)],tmp*0.8,'k-x'); hold on; 
          lb = -prior.std(kk) +prior.mu(kk); ub = prior.std(kk)+ prior.mu(kk);
      else
          lb = prior.lb(kk);       ub = prior.ub(kk);
      end
      plot([lb,lb],tmp*0.5,'k-x'); plot([ub,ub],tmp*0.5,'k-x');
      xlb = prior.lb(kk); xub = prior.ub(kk); cc= xub-xlb;  
      xlim([xlb-cc/5 xub+cc/5]);  
  end
  title(titl{kk},'FontSize',12);
end

subplot(K,1,1);  % title(titl,'FontSize',16);
if exist('prior','var')
    if prior.flag==0
        legend({'Posterior','True value','Posterior mean','Prior mean/std'},...
            'FontSize',12,'Location','northwest'); legend('boxoff');
    else
        legend({'Posterior','True value','Prior bounds'},...
            'FontSize',12,'Location','northwest'); legend('boxoff');
    end
else
    legend({'Posterior','True value','Posterior mean'},...
       'FontSize',12,'Location','northwest'); legend('boxoff');
end
tightfig;


