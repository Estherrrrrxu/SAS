
function plotTrajEns(Usample, nodeID,tN1,sMean,sStd,Utrue,t_ind,errRel, uobs)
% 
[K,tN,M] = size(Usample); ensInd = 1:4:M; 
ensemble = reshape(Usample(nodeID,:,ensInd), tN1,[]);
h= plot(t_ind,ensemble(t_ind,:),'cyan','linewidth',0.5);hold on;
% [Nh, ~] = size(h);  for i=1:Nh;    h(i).Color(4) = 0.5; end  %% transparancy, need < 0.4 to see the difference
h1 = plot(t_ind,sMean(nodeID,t_ind),'b-d','linewidth',1); hold on;
h2 = plot(t_ind,Utrue(nodeID,t_ind),'k-*','linewidth',1);
h3 = plot(t_ind,sMean(nodeID,t_ind)+sStd(nodeID,t_ind),'m-.','linewidth',1); hold on;
h4 = plot(t_ind,sMean(nodeID,t_ind)-sStd(nodeID,t_ind),'m-.','linewidth',1); hold on;
if exist('uobs','var')
    h5 = plot(t_ind,uobs(nodeID,t_ind),'ro','Markersize',6,'linewidth',1); hold on;
    legend([h1,h2,h5],{'Mean','True trajectories','Observations'}, 'Fontsize',12);
else
    legend([h1,h2],{'Mean','True trajectories'}, 'Fontsize',12);
end
xlabel('Time steps'); 
titl = sprintf('Sample trajectories of node %i. Relative error of Mean = %1.3f ',nodeID, errRel(nodeID));
title( titl); setPlot_fontSize; tightfig;  
return

