function [X,V]=StateModel(X0,V0,paraSet,Phi_n)
% StateModel   the non-Markovian statemodel
%              X(n+1)= Phi(n) + V(n)
% where Phi(n) is computed using functionn Phi_n.
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
Rx = length(X0(:,1));
Lx = length(X0(1,:));    
Lv = length(V0(1,:));    
 if min(Lx,Lv)<mLag
     disp('initial data should have length > max(p,q)!'); 
     return;
 else
     XX=X0(:,Lx-mLag+1:Lx);
     VV=V0(:,Lv-mLag+1:Lv); 
 end

X=zeros(Rx,tN);  V=randn(Rx,tN)*sigmaV; 
X(:,1:mLag)=XX; V(:,1:mLag)=VV; 

for i=mLag+1:tN
    X(:,i) = Phi_n(XX,VV,i)+V(i);
    XX = X(:,i-mLag+1:i); 
    VV = V(:,i-mLag+1:i);
end
%  plot(X(1,:));  
return
