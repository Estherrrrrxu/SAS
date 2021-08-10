function plotTrajpf(particles,xloc,titl)
% plot trajectory of the ensemble in SMC
[sN,tN,pN] = size(particles);  % space, time and particle numbers
for p=1:pN
    plot(particles(xloc,:,p)); hold on
end
title(titl); xlabel('time');

return