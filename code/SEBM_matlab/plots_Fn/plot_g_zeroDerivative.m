
function plot_g_zeroDerivative(sample_post,prior,thetaTrue)
% plot the distribution of the zero of g and its derivative at the zero
[K,Nens] = size(sample_post); 
sample_prior = zeros(K,Nens);

if prior.flag ==0     % Gaussian
    sample_prior  = prior.mu + diag(prior.std)*randn(K,Nens);     
elseif prior.flag ==2     % uniform
    sample_prior = prior.lb +diag(prior.ub-prior.lb)*rand(K,Nens); 
end

g_zeros_post   = zeros(1,Nens); 
g_zeros_prior  = zeros(1,Nens); 
g_deriv_post   = zeros(1,Nens);
g_deriv_prior  = zeros(1,Nens); 

for n=1:Nens
     [rootVal,deriv] = zero_derivative(sample_post(:,n));
     g_zeros_post(1,n) = rootVal;
     g_deriv_post(1,n) = deriv;
     [rootVal,deriv] = zero_derivative(sample_prior(:,n));
     g_zeros_prior(1,n) = rootVal;
     g_deriv_prior(1,n) = deriv; 
end

thetaPostMean = mean(sample_post,2);
[meanUe,MeanderivUe] = zero_derivative(thetaPostMean);
[trueUe,derivUe]     = zero_derivative(thetaTrue);

figure; set(gcf, 'Position',  [100, 1000, 500, 300]);
subplot(121)
h1  = histogram(g_zeros_post,20,'Normalization','probability'); hold on 
h2  = histogram(g_zeros_prior,20,'Normalization','probability');
tmp = get(gca,'ylim'); tmp = [0,tmp(2)]; plot([trueUe trueUe],tmp*1.05,'k-*');
hold on;  plot([meanUe meanUe],tmp*1.05,'k-d');
legend('Posterior','Prior','True');  xlabel('Equilibrium state u_e'); ylabel('Probability');
setPlot_fontSize; 

subplot(122)
h1 = histogram(g_deriv_post,20,'Normalization','probability'); hold on
h2 = histogram(g_deriv_prior,20,'Normalization','probability');
tmp = get(gca,'ylim'); tmp = [0,tmp(2)]; plot([derivUe derivUe],tmp*1.05,'k-*');
hold on;  plot([MeanderivUe MeanderivUe],tmp*1.05,'k-d');
legend('Posterior','Prior','True'); 
xlabel('$\frac{dg_\theta}{du}(u_e)$','Interpreter','latex'); ylabel('Probability');
setPlot_fontSize; 
  
%{
figure;
subplot(1,2,1); 
  minPost  = min(g_zeros_post);  maxPost  = max(g_zeros_post);
  minPrior = min(g_zeros_prior); maxPrior = max(g_zeros_prior);
  maxS  = max(maxPost,maxPrior); minS = min(minPost,minPrior);
  gap   =(maxS-minS)/100;   edges = minS:gap:maxS;  
  [f_post, xi1]  = ksdensity(g_zeros_post, edges);  
  [f_prior,xi2] = ksdensity(g_zeros_prior, edges);  
  plot(xi1,f_post,'b-.',xi2,f_prior);  
  xlabel('Equilibrium point u_e'); ylabel('PDF');
  setPlot_fontSize; 
subplot(1,2,2); 
  minPost  = min(g_deriv_post);  maxPost  = max(g_deriv_post);
  minPrior = min(g_deriv_prior); maxPrior = max(g_deriv_prior); 
  maxS     = max(maxPost,maxPrior); minS  = min(minPost,minPrior);
  gap      =(maxS-minS)/100;        edges = minS:gap:maxS;  
  [f_post, xi1]  = ksdensity(g_deriv_post, edges);  
  [f_prior,xi2] = ksdensity(g_deriv_prior, edges);  
  plot(xi1,f_post,xi2,f_prior); 
  xlabel('$\frac{dg}{du}(u_e)$','Interpreter','latex'); ylabel('PDF');
  setPlot_fontSize; 
%}
return


function [rootVal,deriv] = zero_derivative(thetaSample)
% zero of the nonlinear function near 1, and its derivative at the zero
    % ff = @(x) nl_fn(thetaSample,x,0);   rootVal = fzero(ff,1);  
    coefPloy = zeros(5,1); coefPloy([5,2,1]) = thetaSample; a = roots(coefPloy);
    a = real(a); ind1 = find(a>0.9 & a<1.15); 
    if length(ind1) ==1;      rootVal = a(ind1); 
    else; fprintf('Multiple roots of g in [0.9,1,15]! \n'); keyboard; end
    poly_deriv = zeros(4,1);   poly_deriv([4,1]) = [4*coefPloy(5), coefPloy(2)];
    deriv      = polyval(poly_deriv,rootVal);
return