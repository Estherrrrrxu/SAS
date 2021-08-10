function Plots_marginalNodes(filename,figname1,datapath1,Obsdatafile1)
%% %% results representation

close all;
%% load data and assign filenames
load(filename);  % load Sampledata_tN100.mat
load(Obsdatafile1);  % load observation

%% ------------------------------------
%% names of figures
fig_theta = strcat(datapath1,'fig_theta',  figname1);
fig_traj  = strcat(datapath1,'fig_trajEns',figname1);
fig_thetaScatter = strcat(datapath1,'fig_thetaScatter',figname1);
fig_traj2 = strcat(datapath1,'fig_trajEns2',figname1);

%{
%------------------------------------
%% theta estimation   
figNum = 10; % plot the marginal posterior of theta
titl    = {'Marginal posterior \theta_0','Marginal posterior \theta_1',...
          'Marginal posterior \theta_4',};        
samples = theta(:,burnin:numMCMC); 
xlabels = {'\theta_0', '\theta_1','\theta_4'}; 

%
figure(figNum); 
 % plotdensity(samples, thetaTrue, titl,xlabels); 
 MAP = plotdensityPrior_curve(samples, thetaTrue, xlabels,prior); tightfig;
 print(fig_theta,'-depsc');

% % scatter plot matrix of theta
% figure(11); lowerTplotmatrix(samples',thetaTrue); setPlot_fontSize; tightfig; 
% myprintPDF(fig_thetaScatter,'-r600'); % print(fig_thetaScatter,'-depsc'); 
%

 plot_g_zeroDerivative(samples,prior,thetaTrue); tightfig; 
 figname = [datapath1,'g_zeroDeriv',figname1]; myprintPDF(figname); %print(figname,'-depsc'); 
keyboard; 
plot_nlfnEnsemble(thetaTrue, samples, MAP); tightfig; 
figname = [datapath1,'fcn_g',figname1]; myprintPDF(figname); %print(figname,'-depsc'); 
%}
%% ------------------------------------
%% State estimation results
sMean = mean(Usample(:,:,burnin:numMCMC), 3);
sStd  = std(Usample(:,:,burnin:numMCMC),0, 3);
if length(sMean(:,1))==4
    Nodesindx = [1,2,7,8];
    errRel = abs( (Utrue(Nodesindx,:)-sMean)./Utrue(Nodesindx,:));
else
    errRel = abs( (Utrue-sMean)./Utrue);
end
errRel= mean(errRel,2);
errRelavg = mean(errRel); 
fprintf('Relative error of state Estimation by mean = %3.2f\n', errRelavg);


if exist('nodes', 'var') ==1 && exist('nodesID', 'var') ==0 
    nodesID = nodes;  % gap = nodes(2)-nodes(1); 
    nodeObsEst = obs'; 
end

 % % ensemble of traj at a node: observed
tN1 = length(Utrue(1,:)); t_ind = 1:max(1,floor(tN1/40)):tN1-1;  nodeID = nodesID(1); 
%
 figure(8); set(gcf, 'Position',  [100, 1000, 500, 300]);
 plotTrajEns(Usample, nodeID,tN1,sMean,sStd,Utrue,t_ind,errRel,obs);
 axis([0,100,0.94,1.10]);
 myprintPDF(fig_traj,'-r600');  % print(fig_traj,'-depsc');
%{
% plot the posterior of a node: observed
fig_stateDensity = strcat(datapath1,'fig_stateObserved',  figname1);
indx     = nodeID; indt = [20,60, 100]; indsample = burnin:numMCMC;
sgtitl     = sprintf('Marginal posterior of state at node %i ',indx(1)); 
titl     = {sprintf('time = %i',indt(1)),sprintf('time = %i',indt(2)),sprintf('time = %i',indt(3))};
samples  = reshape( Usample(indx,indt, indsample), [],length(indsample)); 
TrueValue= reshape(Utrue(indx,indt),[],1); 
figure; set(gcf, 'Position',  [100, 1000, 500, 300]);
 axisSize =[0.94,1.08,0,0.60]; % Gaussain
 axisSize =[0.94,1.08,0,0.20]; % uniform
plotdensityState(samples, TrueValue, titl, axisSize); tightfig; 
myprintPDF(fig_stateDensity);   %  print(fig_stateDensity,'-depsc');
%}

% % ensemble of traj at an unobserved node
t_ind = 1:max(1,floor(tN1/40)):tN1-1;   nodeID = nodesID(1)+1; 
%
 figure; set(gcf, 'Position',  [100, 1000, 500, 300]);
 plotTrajEns(Usample, nodeID,tN1,sMean,sStd,Utrue,t_ind,errRel);
 myprintPDF( fig_traj2,'-r600'); % print(fig_traj2,'-depsc');
%{
% % % plot the posterior of the unobserved node: 
fig_stateDensity = strcat(datapath1,'fig_stateUnobserved',  figname1);
indx     = nodeID; indt = [20,60, 100]; indsample = burnin:numMCMC;
sgtitl     = sprintf('Marginal posterior of state at node %i ',indx(1)); 
titl     = {sprintf('time = %i',indt(1)),sprintf('time = %i',indt(2)),sprintf('time = %i',indt(3))};
samples  = reshape( Usample(indx,indt, indsample), [],length(indsample)); 
TrueValue= reshape(Utrue(indx,indt),[],1); 
figure;   set(gcf, 'Position',  [100, 1000, 500, 300]);
 axisSize =[0.85,1.15,0,0.54];   % Gaussain
%  axisSize =[0.85,1.15,0,0.20]; % uniform
plotdensityState(samples, TrueValue, titl, axisSize); tightfig; 
myprintPDF(fig_stateDensity);  % print(fig_stateDensity,'-depsc');
%}
%{
% % scatter plot matrix of states
figure; 
lowerTplotmatrix(samples',TrueValue); tightfig; 
print(fig_stateScatter,'-depsc'); 
 
%}
return







