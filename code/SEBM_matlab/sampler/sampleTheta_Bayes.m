%-------------------------------------------------------------------
function [thetaP,svdc1inv,flag] = sampleTheta_Bayes(U,femPar,K,dt, prior,progressON)
% sample the posterior of theta; 
% Because of linear dependence on theta, the likelihoood is computed
% analytically;  thetaP:   1xK
% --- Last updated by Fei Lu, 2019/1/22
% Changes: 
%   - removed Fisher matrix svd truncation (svdc1inv); 2019-1-22. 
%     + this truncation caused bias in MLE; 
%
% % The state model 
%        M2 * U(n+1)  = M*Un + nl(Un,theta) + sqrt(dt)*stdF*pre_chol\N(0,Id)      
%        nl(Un,theta) = theta*terms(Un) ***factors
%               terms = [ones(size(u)); u; u.^4]; 
% % Sn ~ S(Un,U(n+1)) = M2 * U(n+1) - M*Un
% % Gn ~ terms

if nargin <5; prior.flag = 0; end   % if no prior, use N(0,I)  

elements3 = femPar.elements3; 
tri_areas = femPar.tri_areas; 
centerht  = femPar.centerheights;
B         = femPar.B; 
ABmat     = femPar.ABmat; 
Prec_chol = femPar.Prec_chol; 
Cinv      = Prec_chol'*Prec_chol;   % covU inverse without dt*stdF^2 

[~,tN] = size(U); 

% if ~exist('prior.regul','var'); prior.regul =1; end 
if prior.regul;  regul = tN;  % regularization scaling if 1; else, prior ***** 
else;            regul = 1;    end

% % % compute the covariance C1 of theta likelihood
%     C1^{-1} = sum_n Gn'* Cinv * Gn,   
%     mu      = C1* sum_n Gn'* Cinv* Sn
c1inv = zeros(K, K);    mu1  = zeros(K,1);     % c1 ~ the likelihood cov

for tt =1:tN-1
    U0     = U(:,tt); U1 = U(:,tt+1);
   [Sn,Gn] = SnGn(B,ABmat,dt,U0,U1,elements3,centerht,tri_areas,K); 
   c1inv   = c1inv + Gn'*Cinv*Gn; 
   mu1     = mu1 + Gn'*Cinv*Sn; 
end
c1inv    = c1inv/(dt*femPar.stdF^2*regul);   mu1 = mu1/(dt*femPar.stdF^2*regul); 
svdc1inv = svd(c1inv)'; % if min(svdc1inv)<1e-3; c1inv = c1inv+ 1e-3*eye(K); end 
mu1      = c1inv\mu1;   
  
% % % 1-step-posterior combine likelihood with prior 
if prior.flag == 0    % Gaussian prior
   covPriorInv = diag(1./(prior.std).^2); 
   covpost      = (covPriorInv+c1inv)\eye(K);  %   disp(svd(covInv)');   
   covInv_chol = chol(covpost);
   muPost      = covpost * (c1inv*mu1+ covPriorInv*prior.mu);
   % theta       = randn(K,1);
   % thetaP      = muPost+ covInv_chol*theta;
   [thetaP, flag]= thetaThreshold(K,muPost,covInv_chol,prior,progressON);  % for test
elseif prior.flag ==2   % uniform prior: sample from truncated normal
    thetaP = IS_Unifprior(prior,mu1,c1inv);
    % %%%   use truncated Gaussian 
%     lb   = Cov1_chol\(prior.lb- mu1); ub =Cov1_chol\(prior.ub -mu1);  
%     temp = trandn(lb,ub);
%     thetaP = mu1+ Cov1_chol*temp;
    if thetaP(1)<0 
       fprintf('theta0 wrong sign\n');        keyboard;
    end
end
end

function sampleoutput = IS_Unifprior(prior,mu,cov)
% importance sampling for the case with Uniform prior
lb     = prior.lb; ub = prior.ub; 
K      = length(lb); 
numS   = 10; 
sample = zeros(K,numS); wts = ones(1,numS); 
for n = 1:numS 
    temp   = rand(K,1).*(ub-lb) + lb;
    sample(:,n) = temp; 
    sample2 = temp - mu;
    temp    = pinv(cov)*sample2;
    wts(n)  = sum(sample2.*temp);
end
    wts     = wts - max(wts);
    wts     = exp(wts);
    wts     = wts/sum(wts);

indres = myResampling(wts); 
sample = sample(:,indres); 
sampleoutput = sample(:,1); 
end

function [thetaP, flag] = thetaThreshold(K,muPost,covInv_chol,prior,progressON)
% generate theta with threshold. For test
flag =0;
theta       = randn(K,1);
thetaP      = muPost+ covInv_chol*theta;

ind1 = intersect(find(thetaP<prior.ub6std), find(thetaP>prior.lb6std));

fullset = find(prior.ub6std==prior.ub6std);
if length(ind1)~=length(fullset)  % a 2nd chance! :)  -- should not be here
    thetaP = muPost+ covInv_chol*randn(K,1);
    ind1   = intersect(find(thetaP<prior.ub6std), find(thetaP>prior.lb6std));
end
if length(ind1)~=length(fullset)
    if progressON==1
        fprintf('Theta out of range! Set as up-lower bound! \n       ');
    end
    indub = find(thetaP>=prior.ub6std);  indlb = find(thetaP<=prior.lb6std);
    thetaP(ind1) = muPost(ind1);
    thetaP(indub)= prior.ub6std(indub);
    thetaP(indlb)= prior.lb6std(indlb);
    flag =1;
end
end


