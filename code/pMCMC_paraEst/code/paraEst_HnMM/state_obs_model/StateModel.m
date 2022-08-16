function [X,V]=StateModel(X0,V0,paraSet,Phi_n)
% StateModel   the non-Markovian statemodel
%           X(n)  = Phi_n(XX, VV,n-1) + V(n)
%           Phi_n = Terms(XX, VV) * coefs + ftn(n-1).      LxM * Mx1
% where XX=X(n-p:n-1), VV=V(n-q:n-1). 
% Input 
%    X0  -  initial segment, X(1:mLag)
%    V0  -  initial noise segment, V(1:mLag)
%    p,q -  AR and MA order
%    N   -  length of data
% Output
%    X   -  the non-Markovian process X(n)
%    V   -  the noise sequence V(n)

% test 
% p=2; q=1; N=10^3; sigmaV=1; mLag=max(p,q); 
% X0=randn(1,mLag); V0=X0*0.1;

sigmaV = paraSet.sigmaV;    %  variance of the observatin noise
p      = paraSet.p;
q      = paraSet.q; 
tN     = paraSet.tN;         % number of forward steps. 

mLag=max(p,q);
[dx, tx] = size(X0); 
[dv, tv] = size(V0);    
if min(tx,tv)<mLag  || tx ~= tv
    disp('Initial data should have SAME length & > max(p,q)!');
    return;
else
    XX = X0(:,tx-mLag+1:tx);
    VV = V0(:,tv-mLag+1:tv);
end

X = zeros(dx,tN);  V = randn(dv,tN)*sigmaV; 
X(:,1:tx)=XX;      V(:,1:tv)=VV; 

for i=tx+1:tN
    X(:,i) = Phi_n(XX,VV,i-1)+V(i);
    XX = X(:,i-mLag+1:i); 
    VV = V(:,i-mLag+1:i);
end
%  plot(X(1,:));  
return
