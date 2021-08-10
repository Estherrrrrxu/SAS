function [U,U2det] = forward_solver_sphere2(tri_areas,centerheights,ABmat,B,Prec_chol,...
                        elements3, theta,dt,tN,U0,stdF) 
% Difference between foward_solver_sphere1.m and foward_solver_sphere2.m: 
%   + the first compute terms*theta in each nl_fn evaluation; 
%   + the 2nd get all terms, then *theta; this is the same as the inference
%   ====>>>> there are big differences in MLE and Fisher matrix
% 
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
% % The nonlinear term: 
% %            nlterm  = cc *dt* nl(U_{n-1})       ---cc = tri_areas
% %                    = dt * 
% %            nl(U_{n-1}) = dt* sum_j tri_areas(j)/6 * g_theta(CenterHights(j,:)*U) 
% Input:
%   Tri_area - area of the triangle 
%   A,B      - stiff and mass matrix
%   C        - diagnal approximation B; diag(<phi_i,1>)
%   Forc_prec-  
%   stdF     - std of force
 
% By Fei Lu, Nils Weitzel,  18/3/29; 
% Last updated by Fei Lu: 2019/1/22

%% the nonlinear term   
% theta = [0,0,1,1,-0.1];  % parameter p0,...,p4 
% nlfn = @(u,t) nl_fn(theta,u,t); 

K = length(theta); 
Nnodes     = length(B(:,1));      % number of nodes= # of rows in coodinates
Nelements  = length(tri_areas(:,1)); % # of elements (triangles)
% Nnodes    = size(coordinates,1);
% Nelements = size(elements3,1); 

 dirichlet = [];      % no boundary conditions on sphere. 
 FreeNodes = setdiff(1:Nnodes,unique(dirichlet)); %%% all are free nodes
% ABmat =  dt*A(FreeNodes,FreeNodes)+ B(FreeNodes,FreeNodes); %solu= ABmat\b  

U     = zeros(Nnodes,tN+1);   
%% Time integration  
U(:,1) = U0;     % Initial Condition
for n = 2:tN+1
%    b = zeros(Nnodes,1);   
%    % % Non-linear forces: at the gravity center of the triangle
%     for j = 1:Nelements
%         nodesEj = elements3(j,:);
%         temp    = centerheights(j,:)*U(nodesEj,n-1);
%         nl      = nlfn(temp,n*dt);  % nolinear term
%         b(nodesEj)= b(nodesEj) + tri_areas(j)/6* dt*nl;
%     end             
    U0 = U(:,n-1); 
    Gn = termsPara(elements3,Nelements,Nnodes,U0,tri_areas,centerheights,K,dt);
    b  = Gn* theta'; 
    % previous timestep
    b = b + B * U0;
    
    %     % Dirichlet conditions    % no Dirichlet condition on sphere
    %     u = sparse(Nnodes,1);
    %     u(unique(dirichlet)) = u_d(coordinates(unique(dirichlet),:),n*dt);
    %     b = b - (dt * A + B) * u;
    
    % Stochastic forcing:
    bstoc = sqrt(dt) * (Prec_chol\randn(Nnodes,1));
    bstoc = stdF*bstoc;
    
    % Computation of the solution
    u = zeros(Nnodes,1);
    if n==2 && nargout ==2      
        U2det =  ABmat\b(FreeNodes); % deterministic part, used in SMC
    end
    b            = b+ bstoc;
    u(FreeNodes) = ABmat\b(FreeNodes);   %% ----- ======
    U(:,n) = u;
end

return



