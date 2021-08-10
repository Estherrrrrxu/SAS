function [obs, Utrue]= generateDataNodes(thetaTrue,obsPar,femPar,Datafilename,saveON,plotON)
% generate simulated data: observe at partial nodes, instead of elements. 
% Last updated: 2019/1/14

obsPar.thetaTrue = thetaTrue;    % 1x3
tN = obsPar.tN;             dt = obsPar.dt; 

%% forward simulation
Prec_chol     = femPar.Prec_chol; 
stdF          = femPar.stdF; 
elements3     = femPar.elements3;
centerheights = femPar.centerheights;
tri_areas     = femPar.tri_areas ;
ABmat         = femPar.ABmat;    %solu= ABmat\b  
A             = femPar.A;       B      = femPar.B; 
Nnodes = length(A(:,1));      % number of nodes= # of rows in coodinates  
U0      = 0.03*randn(Nnodes,1) + ones(Nnodes,1); 
[Utrue,~] = forward_solver_sphere2(tri_areas,centerheights,ABmat,B,Prec_chol,...
                        elements3, thetaTrue,dt,tN,U0,stdF);          

% % plots the solution
if exist('plotON', 'var') && plotON==1 
    figure;   coordinates = femPar.coordinates; 
    for tt=1:1:length(Utrue(1,:))
        showSphere(elements3,coordinates,full(Utrue(:,tt)), tt);   pause(0.1)
    end
end

% % 2. Animated gif plot
% tt= 1:5:N;    data= full(Utrue(:,tt));
% AnimateGif(elements3,coordinates,data); 

%% generate noisy observation at the time intervals + regions
stdObs  = obsPar.stdObs; % std of observation noise 
  
% % % % Noisy bservation 
nodes   = obsPar.nodes;  % index of observed nodes
[F ]    = Utrue(nodes,2:end);  % F size= [length(nodes),tN-1] 
obs     = F + stdObs* randn(size(F)); 

%% save data
if saveON ==1
    save(Datafilename,  'obs','obsPar','femPar','thetaTrue','Utrue','nodes');
end

return




