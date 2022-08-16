function [Xtrue,obs] = generate_data(para,Phi_n,ploton)

sigmaV = para.sigmaV;    %  variance of the observatin noise
sigmaW = para.sigmaW; 
p      = para.p;
q      = para.q; 

K = 1;      % dimension of the state vector
mLag = max(p,q);
 
X0    = randn(K,mLag); V0 = randn(size(X0))*sigmaV; 
Xtrue = StateModel(X0,V0,para,Phi_n);
noise = randn(size(Xtrue)) .* sigmaW; 
obs   = fnObs_nl(Xtrue) + noise; % nonlinear noisy observations
if ploton
figure; plot(Xtrue(1,:)); hold on; plot(obs,'o-'); 
        legend('Xtrue','Observation');
end
return

