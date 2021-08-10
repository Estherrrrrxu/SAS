% sample the posterior of the states and the parameters
%  --- Method: particle Gibbs with Ancestor Sampling (PGAS)
%  --- only considers parameters in the nonlniear term;
%      the variance of the noise in the SEB is assumed known
% Observation are partial nodes, NOT partial elments!!!
% Last updated: Fei Lu, 2019-8-9

close all;   clear all; 
restoredefaultpath;   addpaths;  
%% basic setttings 
settings_ALL;    % set prior, sampler, observation, and FEM   

stateRegul = 1;  % regulate the posterior of the states by 
if stateRegul==1; SR = 'SR_'; else; SR = ''; end 
%% test traj stability: skip 
%
plotON = 0 ;  saveON = 0; thetaTrue = sampleThetaTrue(prior); obsPar.tN = 10000;
plot_g(thetaTrue,[-5,5],'b-',1,'fig_g');
 [obs,Utrue] = generateDataNodes(prior.mu',obsPar,femPar,'a',saveON,plotON);  
plot(Utrue(3,:)); temp = [mean(Utrue(3,:)), std(Utrue(3,:))];
fprintf('[Mean, Std] of priorMean u(3,t): %2.4f %2.4f\n',temp) ; 
%} 
 shift =''; 
% shiftUD = 0*[-1,-1,-0.3];
% shift = 'shift000'; shiftUD = [-1,-1,-0.3];
% shift = 'shift010'; shiftUD = [-1,1,-0.3]; 
% shift = 'shift011'; shiftUD = [-1,1,0.3];
% shift = 'shift100'; shiftUD = [1,-1,-0.3]; 
% shift = 'shift101'; shiftUD = [1,-1,0.3]
% shift = 'shift110'; shiftUD = [1,1,-0.3]; 
% shift = 'shift111'; shiftUD = [1,1,0.3]

%% output file names: 
if prior.flag ==0;         priorName = 'GaussPrior/'; str1 = '_Gauss'; % 0 Gaussian, 2 uniform
elseif prior.flag == 2;    priorName = 'UnifPrior/';  str1 = '_unif';   end    
nodeNum = sprintf('nodes%i_',length(obsPar.nodes)); nodeNum = [nodeNum,priorName]; 
% if obsPar.stateDim ==42;  
if strcmp( pmcmcPar.sampler, 'IPF')
      datapath = [home_path, 'output/nlfnDeg014_IPF/',nodeNum]; 
else; datapath = [home_path, 'output/nlfnDeg014_SIR/',nodeNum]; 
end
addpath(datapath);


if pmcmcPar.numMCMC == 1000; type = 'nMC1k'; else;  type = 'nMC10k'; end
if pmcmcPar.pgas ==1; cpfas = '_cpfas.mat'; else;  cpfas = '_cpf.mat'; end
sp = pmcmcPar.sp;   % 0 estimate para+states; 1, estimate states with true para
if sp ==1; type1 = strcat('State', type);  % estimate state, uring TRUE/Estimated paramter    
else;      type1 = [type,shift];  % estimate state and paramter  
end

sampleFilename = strcat(datapath,[SR,'Sample_tN_RRR'],num2str(tN), type1,cpfas);
figname        = [strcat('tN',num2str(tN),type1),str1];     
Obsdatafile    = strcat(datapath,'ObsData_tN',num2str(tN),shift,'.mat');

% prior1= prior;  load(sampleFilename);
% prior = prior1; save(sampleFilename,'prior','-append');

%%  Generate observation data and Sample the posterior by particle MCMC: PGAS
if exist(sampleFilename,'file') ~= 2  % if no sample file, generate samples
    % generate Observation data
    if exist(Obsdatafile,'file') == 0
         plotON = 0 ;  saveON = 1;  rng(120)
        % thetaTrue = sampleThetaTrue(prior);  
         thetaTrue = [30.0374  -23.9608   -5.6729]; thetaTrue = prior.mu'+shiftUD;  
        [obs,Utrue] = generateDataNodes(thetaTrue,obsPar,femPar,Obsdatafile,saveON,plotON);  
    end
    load(Obsdatafile); % load Obsdata_tN100.mat; 
    
   % MLE from noisy data ?  NO MLE, because of incomplete data
   % Sample the posterior by particle MCMC: PGAS
    Np      = pmcmcPar.Np;         % number of particles in SMC
    numMCMC = pmcmcPar.numMCMC;  % length of MCMC chain
    burnin  = .3*numMCMC;
    % Run the algorithms:   size(Usample)=[ Dx,tN,numMCMC]; theta: [K,numMCMC]
    fprintf('Running PGAS=%i (N=%i). Progress: ',pmcmcPar.pgas,Np); tic;
     progressON =1; 
     if stateRegul
         [Usample,ess,theta] = pgas_stateParNodes_statePrior(pmcmcPar,obs,obsPar,femPar,prior,progressON);
     else
         [Usample,ess,theta] = pgas_stateParNodes(pmcmcPar,obs,obsPar,femPar,prior,progressON);
     end
                 % size(theta)=K x numMCMC 
    timeelapsed = toc;
    fprintf(' Elapsed time: %2.2f sec.\n\n',timeelapsed); 
        
    varObs  = obsPar.stdObs^2;   dt = obsPar.dt; 
%    clearvars -except timeelapsed Usample ess theta numMCMC Np dt burnin prior pmcmcPar...
%                      thetaTrue Utrue sampleFilename figname; 
   if obsPar.tN*numMCMC>1e6
       Nodesindx = [1,2,7,8]; Usample = Usample(Nodesindx,:,:);
   end
    save dataTemp.mat  timeelapsed Usample ess theta numMCMC Np dt burnin ...
                       prior pmcmcPar thetaTrue Utrue;           
    movefile('dataTemp.mat', sampleFilename);
end

%% present results
% if pmcmcPar.numMCMC ==10000;  plot_thetaMargin_MCMC(sampleFilename,datapath); end 
% load(Obsdatafile);   udensity = uStatFromData(obs,Utrue,figname,datapath, 1); 
%
% load(sampleFilename);
 %{
 figname_state = ['state',figname];  plotON = 1; 
 [crps,es,crps_all,es_all]  = probab_scores(Usample,Utrue,figname_state,datapath,plotON); 
 figname_theta = ['theta',figname];
 [crps,es,crps_all,es_all] = probab_scores(theta,thetaTrue',figname_theta,datapath,plotON);
%}
%{
samples = theta(:,burnin:numMCMC); 
thetaCov = cov(samples');        fprintf('thetaCov \n'); disp(thetaCov); 
thetaStd = sqrt(diag(thetaCov)); fprintf('thetaStd'); disp(thetaStd');
thetaCor = corrcoef(samples');   fprintf('thetaCor\n'); disp(thetaCor);
plotUtrue(Utrue); tightfig; print([datapath,'fig_Utrue'],'-depsc');
%}
% plot_corMCMC(Usample,theta,figname,datapath); 
%{
% coverage frequency: good for the highD state
quantile = 0.90;   
covfreq  = coverageFreq_updateRate(sampleFilename,quantile,figname,datapath);
%}
% plot the marginal of the posterior, and trajectories: of para and state
printFig = 1; 
Plots_marginalNodes(sampleFilename,figname,datapath,Obsdatafile);




