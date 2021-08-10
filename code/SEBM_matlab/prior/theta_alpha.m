function [thetatrue,stdF, thetaLo, thetaUp, thetaStd] =theta_alpha(nldeg)
% scaling: change of variable to obtain dimensionless equation
% from theta (with units) to alpha -- to be used in equation
% 
% Output: thetaTrue, stdF, thetaLo,thetaUp 

theta= [2.883*1e-4, -8.196*1e-7, -7.835*1e-15]/1.0542;
stdF = 1.6639*1e-3; 


c1= 287; c2= 3.14496*1e7; scale_factor= [c2/c1, c2, c1^3*c2];  
alpha  = scale_factor.*theta;
sigmaW = sqrt(c2)/c1 * stdF; 

theta_upper = [3.133425*1e-4, -7.610*1e-7, -6.804*1e-15]/1.0542;
theta_lower = [2.659438*1e-4, -8.533*1e-7, -8.505*1e-15]/1.0542;
alpha_upper = scale_factor.*theta_upper;
alpha_lower = scale_factor.*theta_lower;

if strcmp(nldeg, '014')
    % Output 3 terms: theta0 + theta1*u - theta2*u^4
    % thetatrue = alpha;  % this is not the average of upper and lower
    thetaLo   = alpha_lower; thetaUp = alpha_upper;
    thetatrue = (thetaUp+thetaLo)/2; 
elseif strcmp(nldeg, '04')
    % Output 2 terms: theta0  - theta2*u^4
    ind2 = 2;   % 2 for u^1; 3 for u^4; ===== ALSO change nl_fn
    scale     = [ 1, -alpha(1)/alpha(ind2)] *1;  thetatrue = alpha([1,ind2]).*scale;
    thetaLo   = alpha_lower([1,ind2]).*scale;    thetaUp   = alpha_upper([1,ind2]).*scale;
end

thetaStd  = (thetaUp-thetaLo)/2 /3;  % 3 std in the bounded range  
stdF      = sigmaW; 

%{
fprintf('theta: '); fprintf('%4.4s   ',theta);  fprintf('\n')

fprintf('alpha: '); fprintf('%4.4s   ',alpha); fprintf('\n')

fprintf('stdF:   '); fprintf('%4.4s   ',stdF);  fprintf('\n')
fprintf('sigmaW: '); fprintf('%4.4s   ',sigmaW);  fprintf('\n')


fprintf('alpha_lower: '); fprintf('%4.4f  & ',alpha_lower);  fprintf('\n')

fprintf('alpha_upper: '); fprintf('%4.4f  & ',alpha_upper); fprintf('\n')
%}
