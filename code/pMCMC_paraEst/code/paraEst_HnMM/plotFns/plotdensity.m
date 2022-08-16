function plotdensity(sampleArray, trueV,titl,figNum)
% plot the marginal densities given by samples
figure(figNum); clf; 
K = length(trueV);
for kk = 1:K
  subplot(K,1,kk); 
  samples = sampleArray(kk,:); 
  smean   = mean(samples); % std1 = std(samples); 
  [f,xi]  = ksdensity(samples);
  plot(xi,f); hold on; 
  tmp=get(gca,'ylim');
  plot([trueV(kk) trueV(kk)],tmp*1.5,'k-*');
  plot([smean smean],tmp*1.5,'k-d');
end

 subplot(K,1,1);  title(titl);
 legend('posterior','true value','mean');