function Index_res = myResampling(weights)
%  Resamping (for SMC)  resample to get new index for the samples
%
% (see "A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian
%       Bayesian Tracking," Arulampalam, Maskell, Gordon and Clapp
%       IEEE Trans. on Signal Processing, Vol. 50, No.2, 2002
%  Input:  
%      weight

% construct the cdf
%      P  - number of samples
P=length(weights);
c = zeros(P,1);
for jj=2:P+1
    c(jj)=c(jj-1)+weights(jj-1);
end, clear jj

% sample it and get the stronger particle more often
ii=1; % initialize
Index_res=zeros(1,P);
u1=rand/P; % intialize
for jj=1:P
    u = u1+(jj-1)/P;
    while   u>c(ii) 
        ii=ii+1;
    end
    Index_res(jj) = ii-1;
end