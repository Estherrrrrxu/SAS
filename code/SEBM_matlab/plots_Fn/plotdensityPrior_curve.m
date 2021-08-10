function MAP = plotdensityPrior_curve(sampleArray, trueV,xlabels,prior)
% plot the marginal densities given by samples

K = length(trueV); MAP = zeros(K,1); 
for kk = 1:K
  subplot(K,1,kk); 
  if exist('prior','var')
      xlb = prior.lb(kk); xub = prior.ub(kk); cc= xub-xlb;
      xlim([xlb-cc/5 xub+cc/5]);
      xlimlb = xlb-cc/5; xlimub = xub+cc/5; 
  end
  samples = sampleArray(kk,:); 
  smean   = mean(samples); % std1 = std(samples); 
  minS    = min(samples); maxS  = max(samples); gap =(maxS-minS)/100; 
  if prior.flag ==0;  edges  = min(minS,xlimlb)-gap:gap:max(maxS,xlimub)+gap;
  else; edges  = minS-gap:gap:maxS+gap; end
  %  edges  = minS-gap:gap:maxS+gap;
  
  % [N,edges] = histcounts(samples,edges); f = N/length(samples); xi = edges(1:end-1);  % should not enforce sum(f) =1! should be sum(f*dx) =1
  [f,xi]  = ksdensity(samples, edges);  
  [fmax,indMax] = max(f); MAP(kk)=xi(indMax); sum(f(1:end-1).*(xi(2:end) - xi(1:end-1)))
  plot(xi,f,'b-','linewidth',1); hold on;   
  tmp = get(gca,'ylim'); tmp = [0,fmax]; % tmp = [0,tmp(2)]; 
  plot([trueV(kk) trueV(kk)],tmp*1.2,'k-*');
  plot([smean smean],tmp*1.2,'k-d'); 
  if exist('prior','var')
      if prior.flag==0
          % plot([prior.mu(kk) prior.mu(kk)],tmp*0.8,'k-x'); hold on;
          % lb = -prior.std(kk) +prior.mu(kk); ub = prior.std(kk)+ prior.mu(kk);
          fi = priorValues(xi,prior.mu(kk),prior.std(kk)); 
          fi = fi/sum(fi(1:end-1).*(xi(2:end) - xi(1:end-1)))*0.95;  % 0.95 is aritifical, should be 1 
          plot(xi,fi,'k-.','linewidth',1); hold on;
      else
          lb = prior.lb(kk);       ub = prior.ub(kk);
          % plot([lb,lb],tmp*0.5,'k-x'); plot([ub,ub],tmp*0.5,'k-x');
          plot(xi,ones(size(xi))/(ub-lb), 'k-.','linewidth',1); hold on; 
      end
  end
  xlim([xlimlb xlimub]);   % title(titl{kk},'FontSize',12);
  xlabel(xlabels{kk},'FontSize',12); % ylabel(['p(',xlabels{kk},')']); 
  h = get(gca, 'xlabel'); oldpos = get(h, 'Position'); 
  if kk==1;     set(h, 'Position', oldpos + [-0.25, 0.06, 0]); 
  elseif kk==2; set(h, 'Position', oldpos + [-0.1, 0.08, 0]); end
  % elseif kk==3; set(h, 'Position', oldpos + [0, 0, 0]); end
  setPlot_fontSize; 
end

subplot(K,1,1);  % title(titl,'FontSize',16);
if exist('prior','var')
    if prior.flag==0
        legend({'Posterior','True value','Posterior mean','Prior'},...
            'FontSize',12,'Location','northwest'); legend('boxoff');
    else
        legend({'Posterior','True value','Posterior mean','Prior'},...
            'FontSize',12,'Location','northwest'); legend('boxoff');
    end
else
    legend({'Posterior','True value','Posterior mean'},...
       'FontSize',12,'Location','northwest'); legend('boxoff');
end



return

function fi = priorValues(xi,mean,std)
% evaluate the values of the prior
x   = (xi - mean)/std;
fi  = exp(-x.^2)/sqrt(2*pi*std^2); 
return