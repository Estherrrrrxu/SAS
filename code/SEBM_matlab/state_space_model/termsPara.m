function Gn = termsPara(elements3,Nelements,Nnodes,U0,tri_areas,centerheights,K,dt) 
% terms in the parametric form in discrete system 
%      theta0*term1 + theta1*term2 + theta4*term3
%    K - dimension of paramters
% Each term is the fucntion 
%     \int g_theta(u)
Gn = zeros(Nnodes,K);
for j = 1:Nelements
    nodesEj    = elements3(j,:);
    temp       = centerheights(j,:)*U0(nodesEj);
    [~, terms0]= nl_fn(zeros(1,K),temp,0);   % size(term0)= [K,1]
    Gn(nodesEj,:)  = Gn(nodesEj,:) + tri_areas(j)/3*terms0';
end
Gn = dt*Gn; 
end