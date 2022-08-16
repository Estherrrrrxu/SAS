% function Plots_marginal(printFig,filename,figname)
%% %% results representation

close all;

downsizeXsample(sampleFilename);        % downsize Xsample if numMCMC or tN are large

paraEst(sampleFilename,figname);   % plots of parameter estimation

stateEst(sampleFilename,figname);  % plots of state estimation



function paraEst(filename,figname)
% plot the resutls of parameter estimation
load(filename);  % load Sampledata_tN100.mat

thetaSample = thetaSample'; 
estInd      = ssmPar.thetaInd; 
thetaTrue   = ssmPar.thetaTrue (estInd); 
%% ------------------------------------
% figure names
fig_theta = strcat('fig_theta',figname);
fig_var   = strcat('fig_var',figname);
fig_chain = strcat('fig_chain',figname); 

% theta estimation   
%if strcmp(ssmPar.thetaV, 'theta') || strcmp(ssmPar.thetaV,'thetaV')
    figNum  = 20;
    titl    = 'Theta posteriors';        % plot the posterior of theta
    samples = thetaSample(estInd,burnin:numMCMC);
    plotdensity(samples, thetaTrue, titl,figNum);
    print(fig_theta,'-depsc');
    
    % % scatter plot matrix of theta
    figure(11); 
    lowerTplotmatrix(samples',thetaTrue);        tightfig;
    fig_thetaScatter = strcat('fig_scatter',figname);
    print(fig_thetaScatter,'-depsc'); 
% end

% variance estimation  
if ~ strcmp(ssmPar.thetaV, 'theta')  
    varTrue = (ssmPar.sigmaV).^2;  % edges = [0,0.005:0.01:0.05,1];
    titl    = 'Posterior of variance of state model noise';       
    samplesV = varV(burnin:numMCMC,:);
    figure;  histogram(samplesV); hold on; tmp=get(gca,'ylim');
    plot([varTrue varTrue],tmp,'k-*'); hold off;    title(titl);
    print(fig_var,'-depsc');
end

figure  % plot the chain of the parmater
[L,N] = size(samples);   
for ll = 1:L
    subplot(L+1,1,ll);      
    plot(burnin:numMCMC, samples(ll,:)); hold on;  
    plot(burnin:numMCMC, thetaTrue(ll)*ones(1,N),'k-.'); hold on; 
    yname = strcat('theta-',num2str(ll)); ylabel(yname);  
    if ll==1; legend('sample chain','true value'); end
    if ll<L; set(gca,'xtick',[]);else; xlabel('Chain Steps');  end 
    ymin = min( min(samples(ll,:)), thetaTrue(ll));
    ymax = max( max(samples(ll,:)), thetaTrue(ll)); dy = 0.1*(ymax-ymin);
    axis([burnin,numMCMC, ymin-dy, ymax+dy ]);
end
subplot(L+1,1,L+1);   % plot sigmaV2 
plot(1:numMCMC, varV');  hold on
plot(1:numMCMC, varTrue*ones(1,numMCMC),'k-.');  
ylabel('varState Noise');  xlabel('Chain Steps');
ymin = min( min(varV), varTrue);
ymax = max( max(varV), varTrue); dy = 0.1*(ymax-ymin);
axis([1,numMCMC, ymin-dy, ymax+dy ]);
print(fig_chain,'-depsc');


% plot the correlation 
fig_acf =strcat('fig_acf',figname);  
samples = [samples; varV(burnin:numMCMC)']; 
samp = bsxfun(@minus, samples, mean(samples,2)); 
[L,N] = size(samp);  % LxN
tlag = floor(N/20); Nlag = N-tlag;  
cor  = zeros(L,tlag+1); 
for tt=1:tlag+1
    temp = samp(:,1:Nlag).*samp(:,tt-1+(1:Nlag)); 
    cor(:,tt) = sum(temp,2); 
end
figure
for ll=1:L
    cor(ll,:) = cor(ll,:)/cor(ll,1); 
    plot(cor(ll,:),'linewidth',1); hold on
end
legend('Parameter a_1', 'Parameter a_2','\sigma_V^2'); legend('boxoff') 
xlabel('Lead time'); ylabel('Correlation'); 
title('Correlation of the chain'); 
print(fig_acf,'-depsc'); 
end


function stateEst(filename,figname)
% % present plots of state estimation

load(filename);  % load Sampledata_tN100.mat

fig_state = strcat('fig_state',figname);
fig_traj  = strcat('fig_trajEns',figname);
fig_jump  = strcat('fig_jump',figname);

tnum = length(Xsample(1,:)); 
% ------------------------------------
figNum  = 9;   % plot density of state at a few times
titl    = 'Posteriors of states at a few times';        % plot the posterior of states
tInd2   = 1:floor(tnum/5):tnum;
samples = Xsample(:, tInd2);           trueV = Xtrue(:,tInd2); 
plotdensity(samples, trueV, titl,figNum);
print(fig_state,'-depsc');


figure(8); % % ensemble of traj 
tind = 1:floor(tnum/50):tnum; 
plot(Xsample(:, tind)','cyan'); hold on;
h1 = plot(sMean(tind),'b-d','linewidth',2); hold on;
h2 = plot(Xtrue(tind),'k-*','linewidth',2);
h3 = plot(sMean(tind)+sStd(tind),'r-.','linewidth',1); hold on;
h4 = plot(sMean(tind)-sStd(tind),'r-.','linewidth',1); hold on;
legend([h1,h2],{'MeanEst','true'},'FontSize',16);
set(gca,'XTickLabel',1:floor(tN/10):tN+1);  xlabel('Time'); 
print(fig_traj,'-depsc');



figure   % plot some trajectories to see jumps
plot(Xtrue,'k','linewidth',1); hold on; 
plot(Xsample(30,:),'c'); hold on; 
plot(Xsample(end-10,:),'b--');  
plot(Xsample(end-50,:),'r-.');  
xlabel('time n'); ylabel('u(n)')
legend( 'true','Uchain nMCMC)','Uchain nMCMC-10','Uchain nMCMC-100'); 
title('Sample trajectories in the Markov Chain');
set(gca,'XTickLabel',1:floor(tN/10):tN+1); 
print(fig_jump,'-depsc');




end


function downsizeXsample(filename)
% if the numMCMC and tN are large, downsize the Xsample
load(filename); 

%% plot about the chain: 
figure;   % calculate the update rate at times
tN    = length(Xsample(1,:)) ;   % size Xsample = [numMCMC,tN]
if tN == ssmPar.tN
    update_rates = zeros(1,tN);
    for j= 1:tN
        diff = Xsample(2:end,j)-Xsample(1:end-1,j);
        update_rates(j) = sum( abs(diff)> 0.0001)/(numMCMC-1);
    end
    plot(1:tN,update_rates);
    xlabel('Time'); ylabel('Update rates');
    title('Update (Acceptance) rate of the chain');
    fig_rate = strcat('fig_rate',figname);
    print(fig_rate,'-depsc');
    
    figure % plot correlation function of the chain
    tnum   = ssmPar.tN;
    tInd    = 1:floor(tnum/10):tnum;
    samples = Xsample(:,tInd);
    cor     = acf(samples', floor(tnum/20));        plot(cor');
    xlabel('Lead time'); ylabel('Correlation');
    title('Correlation of the chain at few times on traj');
    fig_cor   = strcat('fig_cor_state',figname);
    print(fig_cor,'-depsc');
end

% % downsize the samples
if ~exist('sMean','var')
    % State estimation
    sMean = mean(Xsample(burnin:numMCMC,:));
    sStd  = std(Xsample(burnin:numMCMC,:),0);
    errRel= norm(Xtrue-sMean)/norm(Xtrue);
    fprintf('Relative error of state Estimation by mean = %3.2f\n', errRel);
    
    if tN >100;  tInd = 1:floor(tN/100):tN; else; tInd = 1:100; end
    if  numMCMC>1000
        numInd  = burnin:floor((numMCMC-burnin)/100):numMCMC;
        Xsample = Xsample(numInd,tInd);  Xtrue   = Xtrue(:,tInd);     
        sMean   = sMean(:,tInd);         sStd    = sStd(:,tInd);
        ess     = ess(numInd,tInd);
    end
    save(filename);
end
end





function cor = acf(samples,tlag)
% compute autocorrelation function
% Input: 
%    samples   - LxN     number of varaibles x observation

samp = bsxfun(@minus, samples, mean(samples,2)); 
[L,N] = size(samp);  % LxN
if exist('tlag','var') ==0;   tlag = floor(N/20);  end
Nlag = N-tlag;
cor  = zeros(L,tlag+1); 
for tt=1:tlag+1
    temp = samp(:,1:Nlag).*samp(:,tt-1+(1:Nlag)); 
    cor(:,tt) = sum(temp,2); 
end
figure
for ll=1:L
    cor(ll,:) = cor(ll,:)/cor(ll,1); 
end
end







