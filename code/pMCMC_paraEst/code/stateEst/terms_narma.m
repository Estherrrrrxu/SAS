function tm =terms_narma(X,V,p,q,meanFlag)
% terms_ARMA:  terms in the ARMA process   
%           X(n+1)=Phi_n + V(n+1),
%           Phi_n = Terms(XX, VV) * coefs.      LxM * Mx1
% where XX=X(n-p:n-1), VV=V(n-q:n-1), and 
%           Terms = (fliplr(XX), fliplr(VV), 1).   LxM    
% ( Then coefs = [ a, c, mu]';     Mx1   )
% Input 
%      X      -  a blocks of AR   KxmLag  
%      V      -  a block of MA.  KxmLag
%  -------------  NOTE: X, V inputs are X(:,n-mLag:n-1)  --------------
%      p,q    -  AR and MA orders for X, V;  
%      meanFlag - 0 if no mu term; 1 if yes. 
% Output
%      tm    - value of the function, a KxM array.

[K,mLag]=size(X); [K1,mLag1]=size(V);
if K~=K1 ||  mLag~=mLag1 
    disp('Wrong size of inputs X and V: they must of the same size!');
    return;
end
X=fliplr(X); V=fliplr(V);
XX=X(:,1:p); 
VV=V(:,1:q);

tm = zeros(K,p+q+meanFlag+1);

tm(:,1:p) = XX;
tm(:,p+1) = XX(:,1)./ (1+ XX(:,1).^2);
tm(:,p+2:p+1+q) = VV; 


return

