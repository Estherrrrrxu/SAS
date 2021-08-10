function [U1,u1det] = forward_ensemble(forward, U0,dt,tN,ind)
% forward an ensemble 
% u1det   - 1-step deterministic forward, U0 NOT resampled by ind
% u1      - tN-step stochastic forward, U0 resampled by ind
% ==== Remark: If tN =1, we can simply add noise to u1det(:,ind) to get U1(:,2,:). 
% However, if tN>1, the multi-step stochastic forward can not be splitted:
%    multi-step (detForward+noise) != multistep detForward + multi-step noise
% Therefore, we simply ran the two separately. 
% === To reduced the cost, one can
%    (1) identify same particle in ind, split-up in one step, but keep the
%    rest steps using stochastic forward
% Last updated: Fei Lu, 2019-1-12 % FL,Nils Weitzel 18-6-14

[dimU,Np] = size(U0);  Nens = length(ind);  
U1    = zeros(dimU,tN+1,Nens);
u1det = zeros(dimU,Np); % deterministic 1-step forward of U0, used in SMC-pgas
  
U0_res=  U0(:,ind);   % U0 resampled by ind
for nn = 1:Np   % should use parfor later
    [~,u1det(:,nn)] = forward(U0(:,nn),dt,1); % 1-step forward to get u1det and
                                              %  the stochastic forward is ingored.
    if nn<= Nens       % tN-step stochastic forward, from U0_res 
        [U1(:,:,nn)] = forward(U0_res(:,nn),dt,tN);   %  u1det not computed, more efficient when no need of u1det
    end
end
   
end