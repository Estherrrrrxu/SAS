function plotdensity(sampleArray, trueV,titl, xlabels)
% plot the marginal densities given by samples

K = length(trueV);
for kk = 1:K
  subplot(K,1,kk); 
  samples = sampleArray(kk,:); 
  smean   = mean(samples); % std1 = std(samples); 
  [f,xi]  = ksdensity(samples);
  plot(xi,f); hold on; 
  tmp=get(gca,'ylim');
  plot([trueV(kk) trueV(kk)],tmp*1.2,'k-*');
  plot([smean smean],tmp*1.2,'k-d');  
  xlabel(xlabels{kk}); 
end

 subplot(K,1,1);  title(titl{1},'FontSize',16);
 legend({'posterior','true value','mean'},'FontSize',16)