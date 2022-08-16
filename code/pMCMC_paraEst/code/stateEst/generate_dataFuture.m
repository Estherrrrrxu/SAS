function [Xtrue,obs,para,Phi_n] = generate_dataNew(para,ploton)
%%% TO be finished
% separate para model and paraPF

a1= 0.9; a2= -0.20; b1=25; d1=0.8; 
coefs = [a1;a2;b1;d1]; 
p=2;q=1;  
fnt =@(n)8*cos(1.2*(n-1) );
para.p = p; 
para.q = q;

sigmaV = para.sigmaV;    %  variance of the observatin noise
sigmaW = para.sigmaW; 



K = 1;      % dimension of the state vector
mLag = max(p,q);
meanFlag = 0; 
% % % Get Phi
Terms = @(XX,VV) terms_narma(XX,VV, p,q,meanFlag);    
Phi_n = @(XX,VV,tn) Terms(XX,VV)*coefs+fnt(tn);    % Phi_n dependes on time

X0    = randn(K,mLag); V0 = randn(size(X0))*sigmaV; 
Xtrue = StateModel(X0,V0,para,Phi_n);
noise = randn(size(Xtrue)) .* sigmaW; 
obs   = fnObs_nl(Xtrue) + noise; % nonlinear noisy observations
if ploton
    figure; plot(Xtrue(1,:)); hold on; plot(obs,'o-');
    legend('Xtrue','Observation');
end
return

