function [obs,femPar, thetaTrue ]= generateData(prior,tN,obs,Datafilename, saveON,numregion, plotON)
% generate simulated data  
% 
% Last updated: 2018/6/6

% thetaTrue = [0.1,-1,0.4,-1,0.2]; % original test
if prior.flag ==1
    temp = (prior.mu)';    % should sample from prior later
    thetaTrue = [exp(temp(1)), temp(2:4), -exp(temp(5))]; % 1x5 vector
elseif prior.flag ==0     % Gaussian
    temp = prior.mu + randn(size(prior.mu)).*sqrt(prior.sigma); thetaTrue = temp'; 
elseif prior.flag ==2     % uniform
    temp = prior.mu +randn(size(prior.mu)).*(prior.ub-prior.lb); thetaTrue = temp'; 
end


dt = 0.01; T = tN*dt;

load mesh_data_tiny.mat; 
% % load  testcase1.mat   regions t0t1 A B Cinv Forc_Prec centerheights coordinates ...
%  %                     K elements3 kappa d tri_areas 
 Prec_chol = chol(Forc_Prec); 
stdF = 0.1; 
 
%% forward simulation
Nnodes  = length(A(:,1));      % number of nodes= # of rows in coodinates  

dirichlet = [];      % no boundary conditions on sphere. 
FreeNodes = setdiff(1:Nnodes,unique(dirichlet)); %%% all are free nodes
ABmat =  dt*A(FreeNodes,FreeNodes)+ B(FreeNodes,FreeNodes); %solu= ABmat\b  

U0      = 0.03*randn(Nnodes,1) + ones(Nnodes,1); 
[Utrue,~] = forward_solver_sphere1(tri_areas,centerheights,ABmat,B,Prec_chol,...
                        elements3, thetaTrue,dt,tN,U0,stdF);          

% % plots the solution
if exist('plotON', 'var') && plotON==1 
    figure;
    for tt=1:1:length(Utrue(1,:))
        showSphere(elements3,coordinates,full(Utrue(:,tt)), tt);   pause(0.05)
    end
end

% % 2. Animated gif plot
% tt= 1:5:N; 
% data= full(Utrue(:,tt));
% AnimateGif(elements3,coordinates,data); 


%% generate noisy observation at the time intervals + regions

  stdObs    = 0.01; % std of observation noise % ======== NILS: 
% stochastic forcing is on the order sqrt(dt)*stdF ~ 0.01; if StdObs is one order higher, 
% it seems impossible for me to infer the true process, as the observation noise cancels
% out the stochastic forcing signal in the process.  
  
% % % % the time intervals and regions of observation 
% t0t1    = [1,2; 2,3; 3,4; 4,6;  6,7; 7,9; 9,10];  % start and end time of each region   tNx2
 t0t1= [(1:tN)',(2:tN+1)']; 
% restrictions on time intervals: ordered, no overlap,
ttN = length(t0t1(:,1));   
rN  = numregion;   % number of regions
eN  = 1;           % number of elements of each region  
regions = randi([1,size(elements3,1)], ttN*rN,eN); % matrix size: [tNxrN, eN]
    %  (# time intervals) x [region1; region2; ...]  
obsPar.t0t1      = t0t1; 
obsPar.regions   = regions;
obsPar.numregion = numregion; 
obsPar.stdObs    = stdObs;


ttN = length(t0t1(:,1));   
rN  = numregion;   % number of regions

    
F  = fnObs(Utrue,t0t1,regions,elements3, tri_areas/3);  %  size= [tN, rN] 

obs = F + stdObs* randn(ttN,rN); 


femPar.Prec_chol = Prec_chol;
femPar.elements3 = elements3;
femPar.tri_areas = tri_areas;
femPar.A         = A; 
femPar.B         = B;
femPar.centerheights = centerheights;
femPar.stdF      = stdF;
forward = @(U0,dt,tn,theta) forward_solver_sphere1(tri_areas,centerheights,ABmat,B,Prec_chol,...
                        elements3, theta,dt,tn,U0,stdF);     
femPar.forward   = forward;
femPar.ABmat     = ABmat; 


%% save data
if saveON ==1
save(Datafilename); 
% save ObsData_tN10_1.mat    A B Prec_chol elements3 tri_areas coordinates kappa ...
%                    centerheights Utrue t0t1 regions obs stdObs rN eN ttN ...
%                    thetaTrue dt T tN femPar; 
end

return

