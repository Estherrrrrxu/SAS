% % % stats_plot: not completed --- 2019-2-10, decided to use table. 
close all; 

path1 = '/Users/feilu/Documents/18/Weitzel/code/BeyesEst_sphere/paraEstSEBnodes/output/nlfnDeg014_IPF/';
 type0 = 'nodes6_UnifPrior';
 % type0 = 'nodes6_GaussPrior'; 

crpstype =1 ; 
if crpstype ==1;     type1  = '/stats/SR_crps_stats_tN100MC10kcpfas.mat';
else;     type1 = '/stats/SR_stats_tN100MC10k_cpfas.mat';         end

filename = [path1,type0,type1]; 
load(filename); 


if crpstype ==1
   temp =cf_state;   meancf_state  = mean(temp);  stdcf_state = std(temp);   % coverage frequence
   temp =crps_state; meancrps_state= mean(temp); stdcrps_state= std(temp);   % Continuous ranked probab score
   temp =es_state;   meanes_state  = mean(temp);  stdes_state = std(temp);   % Energy score
   
   fprintf(' state:     coverageFreq       crps         , Engergy score \n')
   fprintf('&  %2.2f $\\pm$ %2.2f  &  %2.2f $\\pm$ %2.2f &  %2.2f $\\pm$ %2.2f  \n',...
        meancf_state,stdcf_state,meancrps_state,stdcrps_state, meanes_state,stdes_state );
    
   temp =cf_theta;   meancf_theta  = mean(temp,2);  stdcf_theta = std(temp,0,2);   % coverage frequence
   temp =crps_theta; meancrps_theta= mean(temp,2); stdcrps_theta= std(temp,0,2);   % Continuous ranked probab score
   temp =es_theta;   meanes_theta  = mean(temp,2);  stdes_theta = std(temp,0,2);   % Energy score
   
fprintf('\n');       
   fprintf('Theta:      coverageFreq     , Engergy score           crps      \n')
   fprintf('&  %2.2f $\\pm$ %2.2f  &  %2.2f $\\pm$ %2.2f &  %2.2f $\\pm$ %2.2f  \n',...
        meancf_theta,stdcf_theta, meanes_theta,stdes_theta, meancrps_theta,stdcrps_theta);
end
fprintf('\n');  fprintf('\n');  


meanPost = mean(errThetaMeanPost);    errStdmp   = std(errThetaMeanPost);  
MAP = mean(errThetaMaxPost);          errStdMAP = std(errThetaMaxPost);  
[meanPost; errStdmp];
[MAP; errStdMAP];

fprintf('Mean Posterior and MAP(bottom)\n');
for i=1:3
fprintf('&  %2.2f $\\pm$ %2.2f  ',meanPost(i),errStdmp(i));
end
fprintf('\n'); 
for i=1:3
fprintf('&  %2.2f $\\pm$ %2.2f  ',MAP(i),errStdMAP(i));
end
%}

keyboard; 

MeanErrStateRel = mean(errRelState); StderrStateRel = std(errRelState);
temp = errTrajMean(:,21); 
MeanErrStatet20 = mean(temp); StderrStatet20 = std(temp);
temp = errTrajMean(:,61); 
MeanErrStatet60 = mean(temp); StderrStatet60 = std(temp);
temp = errTrajMean(:,101); 
MeanErrStatet100 = mean(temp); StderrStatet100 = std(temp);

fprintf('\n'); 
fprintf('& %2.2f $\\pm$ %2.2f  ',100*MeanErrStateRel,100*StderrStateRel);
fprintf('& %2.2f $\\pm$ %2.2f  ',100*MeanErrStatet20, 100*StderrStatet20);
fprintf('& %2.2f $\\pm$ %2.2f  ',100*MeanErrStatet60, 100*StderrStatet60);
fprintf('& %2.2f $\\pm$ %2.2f  \n',100*MeanErrStatet100, 100*StderrStatet100);

disp([MeanErrStateRel,StderrStateRel; 
    MeanErrStatet20, StderrStatet20;
    MeanErrStatet60, StderrStatet60;
    MeanErrStatet100, StderrStatet100]); 



mkSz =12; ftSz = 14; 

xticks  = {'\theta_0','\theta_1','\theta_4'}; 
boxplot(errThetaMeanPost,'Labels',xticks,'Whisker',1)
title('Compare Random Data from Different Distributions'); % setPlot_fontSize; 

% plotStdErr(type0, meanPost,MAP, xticks, mkSz, ftSz); NOT finished



% MLE from true data
axisbds = [tExp(1)-0.5,tExp(end)+0.5,8,11];
plotcondiNum(svdTrue(:,:,tInd),tExp,type, mkSz, ftSz,axisbds);
axisbds = [tExp(1)-0.5,tExp(end)+0.5,-60,60; 
           tExp(1)-0.5,tExp(end)+0.5,-70,80;
           tExp(1)-0.5,tExp(end)+0.5,-20,20];
plotErrMLE(mleTrue(:,:,tInd),theta_true,tExp,type, mkSz, ftSz,axisbds);

% MLE from Noisy data
axisbds = [tExp(1)-0.5,tExp(end)+0.5,8.2,9.8];
plotcondiNum(svdObs(:,:,tInd),tExp,typeObs, mkSz, ftSz,axisbds);
axisbds = [tExp(1)-0.5,tExp(end)+0.5,-1600,1500; 
           tExp(1)-0.5,tExp(end)+0.5,-1500,2000;
           tExp(1)-0.5,tExp(end)+0.5,-600,400];
plotErrMLE(mleObs(:,:,tInd),theta_true,tExp,typeObs, mkSz, ftSz,axisbds);




function plotStdErr(type, meanPost, MAP, xtick, mkSz, ftSz) % NOT finished
figcondi = ['fig_condiN_',type];

[K,Nsimul,Nt] = size(estimator);  
% condition number
svdmax = reshape(estimator(1,:,:), [Nsimul,Nt]); 
svdmin = reshape(estimator(3,:,:), [Nsimul,Nt]);
Log10condiNum     = log10(svdmax./svdmin); 
cNmean = mean(Log10condiNum,1); 
cNstd  = std(Log10condiNum,0,1); 

figure
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