function corMCMC= MCMCcor(sample, plotON,name)
% MCMC time correlation of samples 
[Dx,N] = size(sample); 
if Dx > 6
    indx = 1:ceil(Dx/6):Dx; 
else 
    indx = 1:Dx;
end
sample  = sample(indx,:); 
Lag     = 100; %max(ceil(N/10),200);
corMCMC = zeros(length(indx),Lag);  % tlag=0:Lag, to get the 0-lag, i.e. covariance 
sample  = sample-mean(sample, 2); 
for ll = 1:Lag
    tlag = ll-1; 
    ind1 = 1:N-tlag;  ind2 = tlag+1:N; 
    temp = sample(:,ind1).*sample(:,ind2); 
    corMCMC(:,ll) = sum(temp,2)/(N-tlag); 
end
corMCMC =corMCMC./corMCMC(:,1);

if plotON==1
    for i=1:Dx
    plot(corMCMC(i,:)); hold on; 
    end
    title([' ',name]); % Correlation of chain: 
    xlabel('Time Lag'); 
end
end