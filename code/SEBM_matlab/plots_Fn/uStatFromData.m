function uObsdensity = uStatFromData(obs,Utrue,figname,datapath, printON)
% get distribution of u from observation data
[Nobs,tN ] = size(obs); 
[Nnodes,~] = size(Utrue); utrue1= reshape(Utrue,1,[]);
obssample  = reshape(obs,1,[]);

rmin       = min(obssample); rmax = max(obssample);
Npartetion = 20;    dr   = (rmax-rmin)/Npartetion; 
edges      = rmin:dr:rmax; 

[fcount,edges] = histcounts(obssample,edges);
binsize = edges(2:end)-edges(1:end-1); 

udistr = fcount/(Nobs*tN);               % prob r in a bin  
udens  = udistr./binsize;              % density rho  = prob/ size(bin)
% plot(edges(1:end-1), udens); 

imin      = find(udens>0,1,'first');
imax      = find(udens>0,1,'last');
udenSupp   = udens(imin:imax);   % rho on support
edgesSupp = edges(imin:imax);   % edges on support of rho

umeanObs  = mean(obssample); ustdObs   = std(obssample); 
umeanTrue = mean(utrue1);    ustdTrue  = std(utrue1); 

uObsdensity.edges     = edges;      
uObsdensity.udens     = udens; 
uObsdensity.edgesSupp = edgesSupp;  
uObsdensity.udenSupp  = udenSupp;  
uObsdensity.umean     = umeanObs;
uObsdensity.ustd      = ustdObs;
uObsdensity.umeanTrue = umeanTrue; 
uObsdensity.ustdTrue  = ustdTrue; 

% plot the 
if printON ==1
    set(gcf, 'Position',  [100, 1000, 400, 400]); 
    histogram(obssample,edges,'Normalization','probability'); hold on;
    histogram(utrue1,edges,'Normalization','probability');
    xlabel('State  variable'); ylabel('Climatological Probability');
    legend({'Observations','True states'},'FontSize', 14); 
    set( gca, 'FontSize', 14); 
    figname = ['solu1_',figname];   figname = [datapath,figname]; tightfig; 
    myprintPDF(figname);
%}

%{
     % h = figure;  % plot the pdf
    [fiObs,xiObs] = ksdensity(obssample,edges); [fi,xi] = ksdensity(utrue1,edges);
    set(gcf, 'Position', [0, 0, 500, 400]);
    subplot(121);   plot(xiObs,fiObs,'b-',xi,fi,'k-.'); hold on;
    xlabel('State variable'); ylabel('Climatological PDF');  legend('Observations','True states'); setPlot_fontSize;
    % plot the histogram 
    subplot(122); histogram(obssample,edges(1:1:end),'Normalization','probability');
    xlabel('State  variable'); ylabel('Climatological Probability');
    legend('Observations'); setPlot_fontSize;
    figname = ['solu_',figname];   figname = [datapath,figname]; tightfig; 
    myprintPDF(figname);
%}
end
return