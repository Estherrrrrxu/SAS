% function plotMLE()
% plot the MLE and the condition number of the Fisher information matrix
close all; 

type = 'Gauss'; 
filename = ['mle_stats_',type,'Prior.mat']; 
load(filename,'svdTrue','mleTrue','svdObs','mleObs','theta_true');

mkSz =12; ftSz = 12; 
tInd = 1:4; tExp = tInd+1; 

%
% Plot error in MLE: mean and std, true and obs, allin1 figure (2subplots)
% axisbds = [tExp(1)-0.5,tExp(end)+1.5,0,2];
% plot_errMLE_allin1(mleTrue,mleObs,theta_true,tExp,type, mkSz,ftSz,axisbds)

% keyboard; 
%}

%
%% ===== plot conditional numbers: true and noisy data
axisbds = [tExp(1)-0.5,tExp(end)+0.5,8,11];
plotcondiNum(svdTrue(:,:,tInd),tExp,type, mkSz, ftSz,axisbds);
typeObs = [type,'_Obs']; 
axisbds = [tExp(1)-0.5,tExp(end)+0.5,8.2,9.8];
plotcondiNum(svdObs(:,:,tInd),tExp,typeObs, mkSz, ftSz,axisbds);
keyboard;
%}


%{
% Plot std of error in MLE: true and obs in 1-figure (2-subplots) -- good
axisbds = [tExp(1)-0.5,tExp(end)+0.5,0,2];
plotStdErrMLE2sets(mleTrue,mleObs,theta_true,tExp,type,mkSz,ftSz,axisbds)
%}
%{
%% ===  Plot std of error in MLE: either true or obs in one figure 
axisbds = [tExp(1)-0.5,tExp(end)+0.5,0,2];
plotStdErrMLE(mleTrue,theta_true,tExp,type, mkSz, ftSz,axisbds) 

typeObs = [type,'_Obs'];
axisbds = [tExp(1)-0.5,tExp(end)+0.5,0.5,3.5];
plotStdErrMLE(mleObs,theta_true,tExp,type, mkSz, ftSz,axisbds)
%}

%{
%% ===== plot conditional numbers: true and noisy data
axisbds = [tExp(1)-0.5,tExp(end)+0.5,8,11];
plotcondiNum(svdTrue(:,:,tInd),tExp,type, mkSz, ftSz,axisbds);
typeObs = [type,'_Obs']; 
axisbds = [tExp(1)-0.5,tExp(end)+0.5,8.2,9.8];
plotcondiNum(svdObs(:,:,tInd),tExp,typeObs, mkSz, ftSz,axisbds);
%}


%{
%% ===== plot MLE from true data and observations
axisbds = [tExp(1)-0.5,tExp(end)+0.5,-60,60; 
           tExp(1)-0.5,tExp(end)+0.5,-70,80;
           tExp(1)-0.5,tExp(end)+0.5,-20,20];
plotErrMLE(mleTrue(:,:,tInd),theta_true,tExp,type, mkSz, ftSz,axisbds);
typeObs = [type,'_Obs'];
axisbds = [tExp(1)-0.5,tExp(end)+0.5,-1600,1500; 
           tExp(1)-0.5,tExp(end)+0.5,-1500,2000;
           tExp(1)-0.5,tExp(end)+0.5,-600,400];
plotErrMLE(mleObs(:,:,tInd),theta_true,tExp,typeObs, mkSz, ftSz,axisbds);
%} 

%{
type = 'UnifPrior';  filename = ['mle_stats_',type,'.mat'];
load(filename,'svdTrue','mleTrue');
svdmax = reshape(svdTrue(1,:,:), [Nsimul,Nt]); 
svdmin = reshape(svdTrue(3,:,:), [Nsimul,Nt]);
Log10condiNum     = log10(svdmax./svdmin); 
cNmean_unif = mean(Log10condiNum,1); 
cNstd_unif  = std(Log10condiNum,0,1);
errorbar(tInd, cNmean_unif, cNstd_unif, 'ks','linewidth',1,'MarkerSize',mkSz); 
axis([1.5,5.5,8.5,11]); xticks(tInd); % axis tight
xlabel('Data size (log_{10}N)','FontSize',ftSz ); ylabel('Condition number (Log10) ','FontSize',ftSz ) 
%}


function plotcondiNum(svdTrue,tExp,type, mkSz, ftSz,axisbds)
figcondi = ['fig_condiN_',type];

[K,Nsimul,Nt] = size(svdTrue);  
% condition number
svdmax = reshape(svdTrue(1,:,:), [Nsimul,Nt]); 
svdmin = reshape(svdTrue(3,:,:), [Nsimul,Nt]);
Log10condiNum     = log10(svdmax./svdmin); 
cNmean = mean(Log10condiNum,1); 
cNstd  = std(Log10condiNum,0,1); 

figure;     set(gcf, 'Position',  [100, 1000, 500, 300]); 
errorbar(tExp, cNmean, cNstd, 'ks','linewidth',1,'MarkerSize',mkSz,'MarkerFaceColor','k'); 
axis(axisbds); xticks(tExp); % axis tight
xlabel('Data size (log_{10}N)','FontSize',ftSz ); ylabel('Condition number (Log10) ','FontSize',ftSz ) 
set(gca,'FontSize',ftSz );       box off;  
print(figcondi, '-depsc');
end


function plotErrMLE(mle,theta_true,tExp,type, mkSz, ftSz,axisbds)
figmle   = ['fig_mle_',type];
[K,Nsimul,Nt] = size(mle);  
% Errors of MLE
errMLEmean = zeros(K,Nt);  errMLEstd  = zeros(K,Nt); 
mle    = mle- theta_true; 
for i=1:K
    temp    = reshape(mle(i,:,:),[Nsimul,Nt]); 
    errMLEmean(i,:) = mean(temp,1); 
    errMLEstd(i,:)  = std(temp,0,1); 
end

figure; clf
subplot(311); errorbar(tExp, errMLEmean(1,:), errMLEstd(1,:), 'ko','linewidth',1,'MarkerSize',mkSz); hold on
    axis(axisbds(1,:)); xticks(tExp); ylabel('error of \theta_0','FontSize',ftSz );
    set(gca,'FontSize',ftSz ); 
subplot(312);errorbar(tExp, errMLEmean(2,:), errMLEstd(2,:), 'k>','linewidth',1,'MarkerSize',mkSz); hold on
    axis(axisbds(2,:)); xticks(tExp); ylabel('error of \theta_1','FontSize',ftSz );
    set(gca,'FontSize',ftSz ); 
subplot(313);errorbar(tExp, errMLEmean(3,:), errMLEstd(3,:), 'k*','linewidth',1,'MarkerSize',mkSz);
    axis(axisbds(3,:)); xticks(tExp); ylabel('error of \theta_4','FontSize',ftSz );
    set(gca,'FontSize',ftSz ); 
xlabel('Data size (log_{10}N)','FontSize',ftSz );    %  box off;  
print(figmle, '-depsc');

end

function plotStdErrMLE(mle,theta_true,tExp,type, mkSz, ftSz,axisbds)
figmle   = ['fig_mleStdErr_',type];
[K,Nsimul,Nt] = size(mle);  
% std of Errors of MLE
Log10errMLEstd  = zeros(K,Nt); 
mle    = mle- theta_true; 
for i=1:K
    temp    = reshape(mle(i,:,:),[Nsimul,Nt]); 
    Log10errMLEstd(i,:)  = log10(std(temp,0,1)); 
end
figure; clf
plot(tExp, Log10errMLEstd(1,:), 'r-o','linewidth',1,'MarkerSize',mkSz); hold on
plot(tExp, Log10errMLEstd(2,:), 'b->','linewidth',1,'MarkerSize',mkSz); hold on
plot(tExp, Log10errMLEstd(3,:), 'k-*','linewidth',1,'MarkerSize',mkSz); hold on
axis(axisbds); xticks(tExp); ylabel('Std of errors (log10)','FontSize',ftSz );
set(gca,'FontSize',ftSz ); 
xlabel('Data size (log_{10}N)','FontSize',ftSz );    %  box off;  
legend('\theta_0','\theta_1','\theta_4');  legend('boxoff')
print(figmle, '-depsc');
end


function plotStdErrMLE2sets(mleTrue,mleObs,theta_true,tExp,type, mkSz,ftSz,axisbds)
figmle   = ['fig_mleStdErr_',type];
[K,Nsimul,Nt] = size(mleTrue);  
% std of Errors of MLE
Log10errStd = zeros(K,Nt); Log10errStd_obs  = zeros(K,Nt); 
mleTrue     = mleTrue - theta_true; 
mleObs      = mleObs  - theta_true; 
for i=1:K
    temp    = reshape(mleTrue(i,:,:),[Nsimul,Nt]); 
    Log10errStd(i,:)  = std(temp,0,1); 
    temp    = reshape(mleObs(i,:,:),[Nsimul,Nt]);    
    Log10errStd_obs(i,:)  = log10(std(temp,0,1)); 
end
figure; clf; 
h1= subplot(211); 
    plot(tExp, Log10errStd(1,:), 'r-o','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, Log10errStd(2,:), 'b->','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, Log10errStd(3,:), 'k-*','linewidth',1,'MarkerSize',mkSz); hold on
    axis(axisbds); xticks(tExp); ylabel('Std of errors (log10)','FontSize',ftSz );
    legend('\theta_0','\theta_1','\theta_4');  legend('boxoff')
    title('Std of errors of MLE from true trajectories')
    set(gca,'FontSize',ftSz ); 
h2= subplot(212); 
    plot(tExp, Log10errStd_obs(1,:), 'r-o','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, Log10errStd_obs(2,:), 'b->','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, Log10errStd_obs(3,:), 'k-*','linewidth',1,'MarkerSize',mkSz); hold on
    axisbds(3) = 0.5; axisbds(4)= 3.5; 
    axis(axisbds); xticks(tExp); ylabel('Std of errors (log10)','FontSize',ftSz );
    legend('\theta_0','\theta_1','\theta_4');  legend('boxoff')
    xlabel('Data size (log_{10}N)','FontSize',ftSz );    %  box off; 
    title('Std of errors of MLE from noisy observations')
    set(gca,'FontSize',ftSz ); 
    
 pos1 = get(h1,'Position'); pos2 = get(h2,'Position'); % [left bottom width height]
 pos2(2) = pos2(2) + 0.02;  set(h2,'Position',pos2); 
     tightfig; print(figmle, '-depsc');
end

function plot_errMLE_allin1(mleTrue,mleObs,theta_true,tExp,type, mkSz,ftSz,axisbds)
figmle   = ['fig_mleErrAllin1_',type];
[K,Nsimul,Nt] = size(mleTrue);  
% std of Errors of MLE
errStd = zeros(K,Nt); errStd_obs  = zeros(K,Nt); 
errMean     = zeros(K,Nt); errMean_obs      = zeros(K,Nt); 
mleTrue     = mleTrue   - theta_true; 
mleObs      = mleObs- theta_true; 
for i=1:K
    temp    = reshape(mleTrue(i,:,:),[Nsimul,Nt]); 
    errStd(i,:)  = std(temp,0,1);
    zz      = mean(temp);  errMean(i,:)       = sign(zz).*abs(zz);
    temp    = reshape(mleObs(i,:,:),[Nsimul,Nt]);    
    errStd_obs(i,:)  = std(temp,0,1); 
    zz      = mean(temp);  errMean_obs(i,:)   = sign(zz).*abs(zz);
end
 figure;  
% h1= subplot(111); 
    plot(tExp, errStd(1,:), 'r-o','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errStd(2,:), 'b->','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errStd(3,:), 'k-*','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errStd_obs(1,:), 'r-.o','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errStd_obs(2,:), 'b-.>','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errStd_obs(3,:), 'k-.*','linewidth',1,'MarkerSize',mkSz); hold on
    axisbds(3) = 10^0; axisbds(4)= 10^3.5;    axis(axisbds); 
    xticks(tExp); ylabel('Std of errors','FontSize',ftSz );
    legend('\theta_0 true-traj.','\theta_1 true-traj.','\theta_4 true-traj.',...
          '\theta_0  noisy-traj.','\theta_1 noisy-traj.','\theta_4 noisy-traj.'); % legend('boxoff'); 
    yscale_symlog();% h1); 
    xlabel('Data size (log_{10}N)','FontSize',ftSz );  
    title('Std of errors of MLE from true and noisy trajectories')
    set(gca,'FontSize',ftSz ); 
    keyboard; 
h2= subplot(212); 
    plot(tExp, errMean(1,:), 'r-o','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errMean(2,:), 'b->','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errMean(3,:), 'k-*','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errMean_obs(1,:), 'r-.o','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errMean_obs(2,:), 'b-.>','linewidth',1,'MarkerSize',mkSz); hold on
    plot(tExp, errMean_obs(3,:), 'k-.*','linewidth',1,'MarkerSize',mkSz); hold on
    xticks(tExp); ylabel('Mean of errors','FontSize',ftSz );
    axisbds(3) = -10^3; axisbds(4)= 2*10^3;        axis(axisbds); 
%    legend('\theta_0 true-traj.','\theta_1 true-traj.','\theta_4 true-traj.',...
%          '\theta_0  noisy-traj.','\theta_1 noisy-traj.','\theta_4 noisy-traj.'); legend('boxoff'); 
    yscale_symlog(h2); 
    xlabel('Data size (log_{10}N)','FontSize',ftSz );    %  box off; 
    title('Mean of errors of MLE from true and noisy trajectories')
    set(gca,'FontSize',ftSz ); 
    
 pos1 = get(h1,'Position'); pos2 = get(h2,'Position'); % [left bottom width height]
 pos2(2) = pos2(2) + 0.02;  set(h2,'Position',pos2); 
     tightfig; print(figmle, '-depsc');
end


function plot_MLEerrbar_allin1(mle,theta_true,tExp,type, mkSz, ftSz,axisbds) % ERROR BAR DOES NOT WORK due to large std
figmle   = ['fig_mle_',type];
[K,Nsimul,Nt] = size(mle);  
% Errors of MLE
errMLEmean = zeros(K,Nt);  errMLEstd  = zeros(K,Nt); 
mle    = mle- theta_true; 
for i=1:K
    temp    = reshape(mle(i,:,:),[Nsimul,Nt]); 
    errMLEmean(i,:) = mean(temp,1); 
    errMLEstd(i,:)  = log10( std(temp,0,1)); 
end

% figure; clf
% subplot(311); errorbar(tExp, errMLEmean(1,:), errMLEstd(1,:), 'ko','linewidth',1,'MarkerSize',mkSz); hold on
%     axis(axisbds(1,:)); xticks(tExp); ylabel('error of \theta_0','FontSize',ftSz );
%     set(gca,'FontSize',ftSz ); 
% subplot(312);errorbar(tExp, errMLEmean(2,:), errMLEstd(2,:), 'k>','linewidth',1,'MarkerSize',mkSz); hold on
%     axis(axisbds(2,:)); xticks(tExp); ylabel('error of \theta_1','FontSize',ftSz );
%     set(gca,'FontSize',ftSz ); 
% subplot(313);errorbar(tExp, errMLEmean(3,:), errMLEstd(3,:), 'k*','linewidth',1,'MarkerSize',mkSz);
%     axis(axisbds(3,:)); xticks(tExp); ylabel('error of \theta_4','FontSize',ftSz );
%     set(gca,'FontSize',ftSz ); 
% xlabel('Data size (log_{10}N)','FontSize',ftSz );    %  box off;  
% print(figmle, '-depsc');

figure; clf; 
h1= subplot(211); 
    errorbar(tExp-0.2, errMLEmean(1,:), errMLEstd(1,:), 'ro','linewidth',1,'MarkerSize',mkSz); hold on
    errorbar(tExp, errMLEmean(2,:), errMLEstd(2,:), 'b>','linewidth',1,'MarkerSize',mkSz); hold on
    errorbar(tExp+0.2, errMLEmean(3,:), errMLEstd(3,:), 'k*','linewidth',1,'MarkerSize',mkSz); hold on
    % axisbds(3) = -8^10^2; axisbds(4)= 8*10^2;    axis(axisbds);  
    xticks(tExp); ylabel('Errors of MLE','FontSize',ftSz );
    legend('\theta_0','\theta_1','\theta_4');  legend('boxoff')
    title('Std of errors of MLE from true trajectories');
    yscale_symlog(h1);     set(gca,'FontSize',ftSz ); 

    
 pos1 = get(h1,'Position'); pos2 = get(h2,'Position'); % [left bottom width height]
 pos2(2) = pos2(2) + 0.02;  set(h2,'Position',pos2); 
     tightfig; print(figmle, '-depsc');

end
