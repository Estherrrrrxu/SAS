
function [f] =fnObs_nl(x)
% The observation function
f  = 1*x.^2/20+ 0*x;  
return