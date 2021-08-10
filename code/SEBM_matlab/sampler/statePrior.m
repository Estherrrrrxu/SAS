function obsPar = statePrior(obsPar,obs)
% estimate prior of states from observation
% use only  mean and variance (set as 2*dataVar to be inclusive)

[Nx,Nt] = size(obs); 
obs1    = reshape(obs,[1,Nx*Nt]);
obsMean = mean(obs1); 
obsStd  = std(obs1) - obsPar.stdObs;  
obsPar.statePriormean = obsMean; 
obsPar.statePriorstd  = 2*obsStd;  % 2*std band to allow for 
end