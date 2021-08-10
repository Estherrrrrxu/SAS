function plotdensityState(sampleArray, trueV,titl, axisSize)
% plot the marginal densities given by samples

K = length(trueV);
for kk = 1:K
  subplot(1,K,kk); 
  samples = sampleArray(kk,:); 
  smean   = mean(samples); % std1 = std(samples); 
  % minS    = min(samples); maxS  = max(samples); gap =(maxS-minS)/1000; 
  % points  = minS-gap:gap:maxS+gap; 
  % [f,xi]  = ksdensity(samples, points);  plot(xi,f); hold on; 
  histogram(samples',20,'Normalization','probability');  hold on
  tmp=get(gca,'ylim'); 
  tmp = [0,min(tmp(2)*1.08,axisSize(4)*0.94)];
  plot([trueV(kk) trueV(kk)],tmp,'k-*'); hold on
  plot([smean smean],tmp,'k-d'); xlabel('State'); 
  if kk==1; ylabel('Probability'); end
  title(titl{kk},'FontSize',12); setPlot_fontSize
  axis(axisSize);
end
% sgtitle(sgtitl);  %only available in MATLAB@R2018b

subplot(1,K,1);  % title(titl,'FontSize',16);
if exist('prior','var')
    if prior.flag==0
        legend({'Posterior','True value','Posterior mean','Prior mean/std'},...
            'FontSize',11,'Location','northwest'); legend('boxoff');
    else
        legend({'Posterior','True value','Prior bounds'},...
            'FontSize',11,'Location','northwest'); legend('boxoff');
    end
else
    h=legend({'Posterior','True','Mean'},...
       'FontSize',11,'Location','northeast'); % legend('boxoff');
    % pos = get(h,'position');
    % set(h, 'position',[0.8 0.5 pos(3:4)])
    set(h,'Position',[0.19 0.789 0.194 0.124]);
end
