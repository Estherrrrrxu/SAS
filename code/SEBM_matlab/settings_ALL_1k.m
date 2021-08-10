
%% settings to be changed in tests; 
SMCsampler = 'IPF' ; % IPF or SIR
numMCMC    = 10000;    % length of Markov chain in pMCMC
pgas       = 1;      % 1 cpf with ancestor sampling; 0 cpf without as  
tN         = 100;    % length of observation trajectory
rlkhd_or_Bayes = 1;  % 1 for regularized likelihood, 0 for Bayes inference    ******


%% settings in the test
nldeg = '014';    % degrees of the nonlinear terms: nl = theta*u^{degree}  
load mesh_data_tiny.mat;   mesh = 'tiny'; 
% % load variabes:   regions t0t1 A B Cinv Forc_Prec centerheights coordinates ...
%  %                     K elements3 kappa d tri_areas 
rng(12); % rng setting


%% ============ prior settings   ================
prior.flag  = 0;  % 0 Gaussian, 2 uniform 
prior.regul = rlkhd_or_Bayes; % 1 for regularized likelihood, 0 for Bayes inference    ******  
[prior,stdF]= setPrior(prior,nldeg);         % ***  This function sets prior: according to prior.flag  ***
%  % size prior.mu=  1xK for nl_fn, otherwise Kx1 (e.g. prior, samples);

dt      = 0.01; 
%% ============ FEM settings   ================
Prec_chol = chol(Forc_Prec);     % Fprc_Prec from mesh, the precision mat
Nnodes    = length(A(:,1));      % number of nodes= # of rows in coodinates  
dirichlet = [];      % no boundary conditions on sphere. 
FreeNodes = setdiff(1:Nnodes,unique(dirichlet)); %%% all are free nodes
ABmat     = dt*A(FreeNodes,FreeNodes)+ B(FreeNodes,FreeNodes); %solu= ABmat\b
 
femPar.Prec_chol = Prec_chol;
femPar.elements3 = elements3;
femPar.tri_areas = tri_areas;
femPar.A         = A; 
femPar.B         = B;
femPar.centerheights = centerheights;
femPar.stdF      = stdF;
forward = @(U0,dt,tn,theta) forward_solver_sphere1(tri_areas,centerheights,...
                   ABmat,B,Prec_chol, elements3, theta,dt,tn,U0,stdF);  
femPar.forward   = forward;
femPar.ABmat     = ABmat; 
femPar.coordinates = coordinates; 
femPar.dt        = dt; 
forward1step     = @(U0,theta) forward_sphere1step(tri_areas,centerheights,...
                   ABmat,B,Prec_chol, elements3, theta,dt,U0,stdF);     
femPar.forward1step  = forward1step;

%% ================  obsevation settings  ================
stdObs    = 0.01; % std of observation noise % 
      %  stochastic forcing is on the order sqrt(dt)*stdF ~ 0.01;  StdObs should be less 
      
t0t1= [(1:tN)',(2:tN+1)'];   % tN comes from the main code
% restrictions on time intervals: ordered, no overlap,
ttN = length(t0t1(:,1));   
numregion = 10;   % number of regions
eN      = 1;          % number of elements of each region  
regions = randi([1,size(elements3,1)], ttN*numregion,eN); 
                  % size: [tNxrN, eN]  (# time intervals) x [region1; region2; ...]
gap     = 6;
nodes   = 1:gap: Nnodes;           % index of observed nodes  ====== v.s. regions
obsPar.t0t1      = t0t1; 
obsPar.regions   = regions;    % size:[ttNxrN,eN](#time intervals) x [region1;region2;...]  
obsPar.numregion = numregion;  % number of regions
obsPar.eleNunReg =  eN;       % number of elements of each region    
obsPar.stdObs    = stdObs; 
obsPar.tN        = tN;          % number of steps for solver and obs
obsPar.dt        = dt; 
obsPar.nodes     = nodes;
obsPar.H         = obsH(nodes,Nnodes); 
obsPar.covObs    = stdObs^2*eye(length(nodes)); 
obsPar.stateDim  = Nnodes;  
 
%% % ================  sampler settings   ================ 
pmcmcPar.sp      = 0; % 0 estimate the parameter; 1, estimate states with true para
pmcmcPar.Np      = 5;       % number of particles in SMC
pmcmcPar.numMCMC = numMCMC; % length of MCMC chain
pmcmcPar.burnin  = 0.3;     % burnin ratio 
pmcmcPar.sampler = SMCsampler;   % 
pmcmcPar.pgas    = pgas;    % 1 cpf with ancestor sampling; 0 cpf without as    

% the matrices for IPF [see bottom]
covU     = (ABmat*Prec_chol)\eye(Nnodes); covU = covU*covU'*dt*stdF^2; 
covY     = eye(length(nodes))*(stdObs^2) + obsPar.H*covU * obsPar.H'; 
Kmat     = covU*obsPar.H'*(covY\eye(length(nodes)));  
pmcmcPar.Kmat          = Kmat;    
pmcmcPar.covprior      = covU;   
pmcmcPar.covpost       = covU - Kmat*obsPar.H*covU;  
pmcmcPar.chol_priorInv = ABmat*Prec_chol /(sqrt(dt)*stdF);  
pmcmcPar.chol_post     = chol(pmcmcPar.covpost);


  
%%  % ================ Print settings: ========= 

if strcmp( pmcmcPar.sampler, 'IPF'); fprintf('  IPF in SMC\n');
else;                                fprintf('  SIR in SMC\n'); end

if  pmcmcPar.pgas==1; fprintf('  CFP with Ancestor Sampling\n');
else;                 fprintf('  CFP without Ancestor Sampling\n'); end

fprintf('observing nodes with gap = %i \n',gap);
% fprintf('\n svd of covariance in 1step state-space model:\n')
% fprintf('    posterior     prior    obs \n'); 
% disp([svd(pmcmcPar.chol_post ), svd(covU), stdObs.^2* ones(Nnodes,1)]);  

fprintf('\n Markov Chain size: %i, # particles= %i\n\n', pmcmcPar.numMCMC,pmcmcPar.Np); 

clear t0t1 regions numregion stdObs dt nodes Nnodes ttN eN gap; 
clear Prec_chol Forc_Prec dirichlet FreeNodes ABmat tri_areas centerheights...
      elements3 coordinates A B Cinv stdF; 
clear SMCsampler numMCMC pgas rlkhd_or_Bayes; 


function H = obsH(nodes, Nnodes)
% the observation matrix
dimObs = length(nodes); 
H = zeros(dimObs,Nnodes);
for m = 1:dimObs
    H(m,nodes(m)) = 1;
end
% H = spones(H); 
end

%% % State model in Gaussian transition format:
%             U_n = bn + sqrt(dt)*stdF*ABmat\Prec_chol\N(0,Id)
%                 = bn + N(0,C_U),    
%  where   bn  = udet from forward_ensemble.m
%          C_U = dt*stdF^2*Cmat*Cmat^T, with Cmat^{-1} = ABmat*Prec_chol
% % Observation model
%             Y_n = H U_n + V_n
% % In optimal importance/Implicit sampling, sample of U_n is dran from
% Gaussian with mean and covariance: 
%             mean = bn + C_U *H'*C_Y^{-1} *(y_n-H*bn)
%             Cov  = C_U - C_U*H'*C_Y^{-1} *H*C_U
% where  C_Y = C_V + H*C_U*H'; 
% In code:    Kmat = C_U *H'*C_Y^{-1}; 
%             mupost  = muprior + Kmat * (yn - bn(nodes))
%             covpost = covU - Kmat*H*C_U
% weight of the particles
%        F(x1,mupior,mupost,yobs) = -1/2 * 
%         ( norm(chol_covU\(x1-muprior),2)^2 + norm(yobs-x1(nodes))^2/stdObs^2
%         +1/2 * (  norm( chol_covpost\(x1 - mupost),2)^2  
%                 + norm( yobs- muprior(nodes),2)^2/stdObs^2 )
%  




