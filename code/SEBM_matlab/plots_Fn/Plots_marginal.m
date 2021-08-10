function Plots_marginal(filename,figname,datapath,Obsdatafile)
%% %% results representation

close all;
%% load data and assign filenames
load(filename);  % load Sampledata_tN100.mat
load(Obsdatafile);  % load observation

%% ------------------------------------
%% plot the figures
fig_theta = strcat(datapath,'fig_theta',  figname);
fig_traj  = strcat(datapath,'fig_trajEns',figname);
fig_thetaScatter = strcat(datapath,'fig_thetaScatter',figname);


%------------------------------------
% theta estimation   
figNum = 10; 
titl    = 'Theta posteriors';        % plot the posterior of theta
samples = theta(:,burnin:numMCMC); 
xlabels = {'\theta_0', '\theta_1','\theta_2'}; 

figure(figNum); 
% plotdensity(samples, thetaTrue, titl,xlabels); 
plotdensityPrior(samples, thetaTrue, titl,prior); 
% thetaM   = mean(theta(:,burnin:numMCMC),2)';  % mean and std of estimator
% thetaStd = std(theta(:,burnin:numMCMC),0,2);
 print(fig_theta,'-depsc');

% % scatter plot matrix of theta
figure(11); 
samples = theta(:,burnin:numMCMC); 
lowerTplotmatrix(samples',thetaTrue);
tightfig; 
print(fig_thetaScatter,'-depsc'); 


%% ------------------------------------
% State estimation 
sMean = mean(Usample(:,:,burnin:numMCMC), 3);
sStd  = std(Usample(:,:,burnin:numMCMC),0, 3);
errRel= norm(Utrue-sMean)/norm(Utrue);
fprintf('Relative error of state Estimation by mean = %3.2f\n', errRel);


if exist('nodes', 'var') ==1 && exist('nodesID', 'var') ==0 
    nodesID = nodes; 
    nodeObsEst = obs'; 
end

    
    
figure(8); % % ensemble of traj at a space point
tN1 = length(Utrue(1,:)); t_ind = 1:max(1,floor(tN1/100)):tN1-1;  nodeID = nodesID(2); 
ensemble = reshape(Usample(nodeID,:,:), tN1,[]); 
plot(ensemble(t_ind,:),'cyan'); hold on;
h1 = plot(sMean(nodeID,t_ind),'b-d','linewidth',1); hold on;
h2 = plot(Utrue(nodeID,t_ind),'k-*','linewidth',1);
h3 = plot(sMean(nodeID,t_ind)+sStd(nodeID,t_ind),'m-.','linewidth',1); hold on;
h4 = plot(sMean(nodeID,t_ind)-sStd(nodeID,t_ind),'m-.','linewidth',1); hold on;
h5 = plot(nodeObsEst(t_ind,2),'ro','linewidth',0.5); hold on;  xlabel('Time steps'); 
legend([h1,h2,h5],{'MeanEst','true','Obs-Est'},'FontSize',16);
titl = ['Ensemble trajectories of Node 3. ', 'Reltive error of MeanEst = ',num2str(errRel)]; 
title( titl); 
print(fig_traj,'-depsc');

figure(1); nodeID = nodesID(3); 
ensemble = reshape(Usample(nodeID,:,:), tN1,[]);
plot(ensemble(t_ind,:),'cyan'); hold on;
h1 = plot(sMean(nodeID,t_ind),'b-d','linewidth',1); hold on;
h2 = plot(Utrue(nodeID,t_ind),'k-*','linewidth',1);
h3 = plot(sMean(nodeID,t_ind)+sStd(nodeID,t_ind),'m-.','linewidth',1); hold on;
h4 = plot(sMean(nodeID,t_ind)-sStd(nodeID,t_ind),'m-.','linewidth',1); hold on;
h5 = plot(nodeObsEst(t_ind,3),'ro','linewidth',0.5); hold on;
xlabel('Time steps'); 
legend([h1,h2,h5],{'MeanEst','true','Obs-average'},'FontSize',16);
titl = ['Ensemble trajectories of Node 6. ', 'Reltive error of MeanEst = ',num2str(errRel)];
title( titl); print('fig_trajNode6','-depsc');


%{
figure(7)   % plot some trajectories to see jumps
spaceind =1; 
plot(Utrue(spaceind,:),'k','linewidth',1); hold on; 
plot(Usample(spaceind,:,end),'c'); hold on; 
plot(Usample(spaceind,:,end-10),'b--');  
plot(Usample(spaceind,:,end-100),'r-.');  
xlabel('time n'); ylabel('u(n)')
legend( 'true','Uchain nMCMC)','Uchain nMCMC-10','Uchain nMCMC-100'); 
title('Sample trajectories in the Markov Chain');
print(fig_jump,'-depsc');
%}


    plotUtrue(Utrue); 
    
return



function plotUtrue(Utrue)
% plot Utrue to see if the parameters are reasonable
figure
subplot(211); plot(Utrue'); title('All nodes');  
subplot(212); mUtrue = mean(Utrue,2);   stdUtrue = std(Utrue,0,2); 
        plot(mUtrue); hold on; plot(mUtrue+stdUtrue); plot(mUtrue-stdUtrue); 
        title('Mean and std of Utrue'); legend('Mean \pm std ')
        print('fig_Utrue','-depsc'); 

return














