function [u1det,U1] = forward_ensemble1step(forward1step, U0,ind)
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
U1    = zeros(dimU,Nens);
u1det = zeros(dimU,Np); % deterministic 1-step forward of U0, used in SMC-pgas

if nargout ==1      %  determinsitic part
    for nn = 1:Np   % should use parfor later
        u1det(:,nn) = forward1step(U0(:,nn));
    end
end
if nargout ==2      %  determinsitic part and with stochastic
    for nn = 1:Np   % should use parfor later
        [u1det(:,nn), U1(:,nn)] = forward1step(U0(:,nn));
    end
    U1 = U1(:,ind);
end

end