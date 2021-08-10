function [mu1,svdc1inv,stdFest] = MLE_truestate2(U,femPar,K) % NOT USED
% MLE from true state data 
% Because of linear dependence on theta, the likelihoood is computed
% analytically;  thetaP:   1xK
% --- Last updated by Fei Lu, 2019/1/20 
%
% % The state model
%         U(n+1)  = ABmat\(B*Un + nl(Un,theta) + sqrt(dt)*stdF*pre_chol\N(0,Id) )     
%                 = ABmat\b + ABmat\ ( sqrt(dt)*stdF*pre_chol\N(0,Id) )  
%        nl(Un,theta) = Gn * theta
% % Sn ~ S(Un,U(n+1)) = ABmat * U(n+1) - B*Un   
% % Gn ~ terms*factors
% %       pre_chol*( Sn -Gn*theta) =sqrt(dt)*stdF* N(0,Id)
% %      Diff  = Sn- Gn*theta 

elements3 = femPar.elements3; 
tri_areas = femPar.tri_areas; 
centerht  = femPar.centerheights;
B         = femPar.B; 
ABmat     = femPar.ABmat; 
Prec_chol = femPar.Prec_chol; 
Cinv      = Prec_chol'*Prec_chol;   % covU inverse without dt*stdF^2 
dt        = femPar.dt; 

[nNodes,tN] = size(U); 

% % % compute the covariance C1 of theta likelihood
%     C1^{-1} = sum_n Gn'* Cinv * Gn,   
%     mu      = C1* sum_n Gn'* Cinv* Sn
c1inv = zeros(K, K);    mu1  = zeros(K,1);     % c1 ~ the likelihood cov
Snpath= zeros(nNodes,tN-1); 
Gnpath= zeros(nNodes,K,tN-1); 
for tt =1:tN-1
    U0     = U(:,tt); U1 = U(:,tt+1);
   [Sn,Gn] = SnGn(B,ABmat,dt,U0,U1,elements3,centerht,tri_areas,K); 
   
   c1inv   = c1inv + Gn'*Cinv*Gn; 
   mu1     = mu1   + Gn'*Cinv*Sn;
   Snpath(:,tt) = Sn; 
   Gnpath(:,:,tt) = Gn; 
end
c1inv    = c1inv/(dt*femPar.stdF^2*tN);   mu1 = mu1/(dt*femPar.stdF^2*tN); 
svdc1inv = svd(c1inv)';
mu1      = c1inv\mu1;   

residual = 0; 
for  tt =1:tN-1
    diffEst  = Snpath(:,tt) - Gnpath(:,:,tt)*mu1;   % size= nNodes x K
    residual = residual+ norm(Prec_chol*diffEst,'fro')^2; 
end

stdFest = sqrt(residual /(nNodes*tN*dt)); 

return

