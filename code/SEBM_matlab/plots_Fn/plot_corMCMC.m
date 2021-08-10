function plot_corMCMC(Usample,thetasample,figname,datapath)
% plot correlation of the chain 
plotON =1;  
% xind  = 1:7:12; 
xind = [1,4];
sampleU = reshape(Usample(xind,[10,40,90],:),[],length(thetasample(1,:)));  
figure; set(gcf, 'Position',  [100, 1000, 500, 300]);
subplot(121); corMCMCstate = MCMCcor(sampleU, plotON,'States');  ylabel('Correlation');
lgnd = {'U_{10,1}','U_{10,8}','U_{40,1}','U_{40,8}','U_{90,1}','U_{90,8}'};legend(lgnd); 
linestyle  = {'-';'-';'--';'--';':';':'};
colorArray = {'b';'k';'b';'k';'b';'k'}; lineW = {1;1;1;1;2;2};
% h_axes = gca; h_plot = get(h_axes,'Children'); ylabel('ACF');
% set(h_plot,{'LineStyle'},linestyle, {'Color'}, colorArray,{'LineWidth'},lineW); 
% xticks([0 50 100 150 200]); 
setPlot; 

subplot(122); corMCMCpara  = MCMCcor(thetasample, plotON,'Parameters');
lgnd = {'\theta_0', '\theta_1','\theta_4'}; legend(lgnd); 
linestyle  = {'-';'--';':'}; colorArray = {'b';'r';'k'}; lineW = {1;1;2};
% h_axes = gca; h_plot = get(h_axes,'Children');
% set(h_plot,{'LineStyle'},linestyle, {'Color'}, colorArray,{'LineWidth'},lineW); 
setPlot; 
fig_corMC  = strcat(datapath,'fig_corMC',figname); tightfig; 
print(fig_corMC,'-depsc');
 
return