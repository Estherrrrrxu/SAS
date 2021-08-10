

function plot_nlfnEnsemble(thetaTrue, samples, MAP)
bds = [0.90,1.15]; 

figure; set(gcf, 'Position',  [100, 1000, 500, 300]);
[K,Nens] = size(samples); 
Nplot = max(100,floor(Nens/10));
ind = randi(Nens,1,Nplot);
for n=1:1:Nplot
    temp = samples(:,ind(n));
    h = plot_g(temp',bds,'cyan-',0.5); hold on;
    h.Color(4) = 0.3;
end
h2 = plot_g(mean(samples,2)',bds, 'b-',1.5); hold on; 
h3 = plot_g(MAP',bds, 'r--',1.5); hold on;
h1 = plot_g(thetaTrue,bds,'k:',2); hold on;
plot_g(0*thetaTrue,bds,'k',0.35); 
legend([h1,h2,h3],{'True','Posterior Mean','MAP'}, 'Fontsize',12);
% title('The nonlinear function g_\theta');
xlabel('x'); ylabel('g_\theta(x)');
axis([0.90,1.15,-10,8]);
% linestyle  = {'-';'--';':'}; colorArray = {'b';'r';'k'}; lineW = {1;1;2};
setPlot_fontSize; 

return

