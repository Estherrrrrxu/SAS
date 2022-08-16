function tm =terms(X,deg)
% terms_ARMA:  terms in the Markov process   
%           X(n+1)=Phi_n + V(n+1),
%           Phi_n = Terms(X) * coefs.         
% where 
%           Terms = [1,x.^(1:deg)]         size = [length(X), deg+1]
% ( Then size coefs = [deg+1,1]      )
% Input 
%      X      - Kx1 
%      dge    - deg of polynomial
% Output
%      tm    - value of the function, a Kx(deg+1) array.

[K,L]=size(X); 

if K==1
    X = X'; K = L; 
end
tm = [ones(K,1), X.^(1:deg)]; 

return

