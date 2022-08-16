
function MLE_trueX(ssmPar)
% Estimate the parameters by MLE from true X and V
% see StateModel for details about the model

K      = 1;  
mLag   = max(ssmPar.p, ssmPar.q); 
sigmaV = ssmPar.sigmaV; 

% ssmPar.tN  = 1000;     % number of time steps in state space model
X0 = randn(K,mLag);   V0 = randn(size(X0))*sigmaV;
[Xtrue, Vtrue] = StateModel(X0,V0,ssmPar,ssmPar.Phi_n);
%  plot(Xtrue); 

% Xtrue =  Xtrue  + 0.2*randn(size(Xtrue)); 

tN    = ssmPar.tN;  Terms = ssmPar.Terms; fnt   = ssmPar.fnt; 
tms = zeros(length(ssmPar.thetaTrue),tN-mLag); 
Xft = zeros(1,tN-mLag); 
for t = mLag+1:tN-1             % estimate the terms in likelihood
    XX = Xtrue(:,t-mLag+1:t); 
    VV = Vtrue(:,t-mLag+1:t);
    tms(:,t) = Terms(XX,VV);
    Xft(:,t) = Xtrue(t+1) - fnt(t);  
end

%% the MLE: Ax = b  
A      = tms * tms';   b  = tms * Xft'; % theta = A\b;       
estInd = ssmPar.thetaInd; knownInd = setdiff(ssmPar.allInd,estInd); 
b     =  b(estInd)- A(estInd,knownInd)*ssmPar.thetaTrue(knownInd);
A     = A(estInd,estInd); 
A     = A/tN;  b  =  b/tN; 

[~,S,~] = svd(A);     S    = sqrt(diag(S)); 
fprintf('\n Eigen-values of the likelihood matrix: \n');
disp(S');

theta = A\b;          %  MLE

thetaEst = ssmPar.thetaTrue; thetaEst(estInd) = theta;
Vest  =Xft - thetaEst'* tms;
sVest = var(Vest); 
sV    = var(Vtrue); 
fprintf('sVest and sV: %4.4f, %4.4f\n',sVest,sV);

disp('True (top row) and MLE coefficients:');
disp([ssmPar.thetaTrue(estInd)'; theta']);  
fprintf('True and MLE sigmaV: %4.2f %4.2f \n', ssmPar.sigmaV, sVest); 
end 

