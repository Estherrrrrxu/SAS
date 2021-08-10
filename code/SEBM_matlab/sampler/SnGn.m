
function  [Sn,Gn] = SnGn(B,ABmat,dt,U0,U1,elements3,centerheights,tri_areas,K)
% the term S(U_n, U_{n+1},\theta) in posterior
% Input
%    M, M1        - mass and stiff matrices
%    U0,U1        - U(:,n-1) and U(:,n), or their vector;  NxtN
%    dt           - time step size
%    elements3    - triangle elements
%    coordinates  - coordiantes of the nodes
%    nlfn         - the nonlinear function  g(u,t)
%    t            - times coresponging to U 
%    K            - dimension of parameter
% Output
%    suu           - value of S(U_n, U_{n+1},\theta)
% % The state model
%        M2 * U(n+1)  = M*Un + nl(Un,theta) + pre_chol\N(0,Id)      
%        nl(Un,theta) = theta*terms(Un) ***factors
%               terms = [ones(size(u)); u; u.^4]; 
% % Sn ~ S(Un,U(n+1)) = M2 * U(n+1) - M*Un
% % Gn ~ terms

% % Non-linear forces: at the gravity center of the triangle
%     for j = 1:Nelements
%         nodesEj = elements3(j,:);
%         temp    = centerheights(j,:)*U(nodesEj,n-1);
%         nl      = nlfn(temp,0);  % nolinear term   ---- nl = theta*terms; 
%         b(nodesEj)= b(nodesEj) + tri_areas(j)/6* dt*nl;
%     end

[Nnodes, ~] = size(U0);
Nelements = size(elements3,1);
Gn = zeros(Nnodes,K);
Sn = ABmat*U1- B*U0;

% for j = 1:Nelements
%     nodesEj    = elements3(j,:);
%     temp       = centerheights(j,:)*U0(nodesEj);
%     [~, terms0]= nl_fn(zeros(1,K),temp,0);   % size(term0)= [K,1]
%     Gn(nodesEj,:)  = Gn(nodesEj,:) + tri_areas(j)/3*terms0';
% end
% Gn = dt*Gn; 
Gn = termsPara(elements3,Nelements,Nnodes,U0,tri_areas,centerheights,K,dt); 

end 

