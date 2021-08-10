function [Udet,Ustoc] = forward_sphere1step_2(tri_areas,centerheights,ABmat,B,Prec_chol,...
                        elements3, theta,dt,U0,stdF)
% 1-step forward  FEM2D_nlSP  
% finite element method for two-dimensional stochastic energy balance model
%(a nonlinear stochastic parabolic equation).
%
%    FEM2D_HEAT solves heat equation 
%           d/dt u =  nu*Delta u +F(u) + f ;   f = Gaussian noise
%                u = u_D           on the Dirichlet boundary
%           d/dn u = g             on the Neumann boundary (=0,not used)   
%    on a geometry described by triangles and presents the solution
%    graphically. 
% 
% Reference: 
%    Remarks around 50 lines of Matlab: Short finite element implementation
%    J. Alberty, C. Carstensen and S. A. Funken.    
%
% % Using semi-backward Euler: 
% %         (dt*A + B)U_n = b + B*U_{n-1} 
% % where      A = <D phi_i,D phi_j>, B = <phi_i,phi_j>
% %            b = cc *dt* nl(U_{n-1}) + sqrt(dt) * stoForce 
% % In Gaussian transition format:
% %            U_n = bn + sqrt(dt)*stdF*ABmat\Prec_chol\N(0,Id)
% Input:
%   Tri_area - area of the triangle 
%   A,B      - stiff and mass matrix
%   C        - diagnal approximation B; diag(<phi_i,1>)
%   Forc_prec-  
%   stdF     - std of force
% Last updated by Fei Lu: 2019/1/12   By Fei Lu, Nils Weitzel, 18/3/29; 

%% the nonlinear term   
% theta = [0,0,1,1,-0.1];  % parameter p0,...,p4 
K = length(theta); 

Nnodes     = length(B(:,1));      % # of nodes       % Nnodes    = size(coordinates,1); 
Nelements  = length(tri_areas(:,1)); % # of elements % Nelements = size(elements3,1); 

dirichlet = [];      % no boundary conditions on sphere. 
FreeNodes = setdiff(1:Nnodes,unique(dirichlet)); %%% all are free nodes
% ABmat =  dt*A(FreeNodes,FreeNodes)+ B(FreeNodes,FreeNodes); %solu= ABmat\b  

% b     = zeros(Nnodes,1);
% % Non-linear forces: at the gravity center of the triangle
% for j = 1:Nelements
%     nodesEj = elements3(j,:);
%     temp    = centerheights(j,:)*U0(nodesEj);
%     nl      = nlfn(temp,1*dt);  % nolinear term
%     b(nodesEj)= b(nodesEj) + tri_areas(j)/6* dt*nl;
% end 
Gn = termsPara(elements3,Nelements,Nnodes,U0,tri_areas,centerheights,K,dt);
b  = Gn* theta'; 

b = b + B * U0;       % previous timestep
%     % Dirichlet conditions    % no Dirichlet condition on sphere
%     u = sparse(Nnodes,1);
%     u(unique(dirichlet)) = u_d(coordinates(unique(dirichlet),:),1*dt);
%     b = b - (dt * A + B) * u;

% Computation deterministic part of the forward
Udet =  ABmat\b(FreeNodes); % deterministic part, used in SMC
% Computation stochastic part of the forward
if  nargout == 2
    bstoc = sqrt(dt) * (Prec_chol\randn(Nnodes,1));  % Stochastic forcing:
    bstoc = stdF*bstoc;     u = zeros(Nnodes,1);
    u(FreeNodes) = ABmat\bstoc(FreeNodes);   %% ----- ======
    Ustoc = Udet+ u;
end

return



