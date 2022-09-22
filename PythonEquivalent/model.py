# %%
# Input the settings of the state space model and the priors
# The state space model: 
#   x_t = x_{t-1} + \Delta t * [J(t) - Q(t) * \Omega_Q(x_{t-1}) - d x_t / d T)],
#   y_t = 
#
#   x_t = S_T at t
#   y_t = \Delta C_Q(t) = C_Q(t) - C_Q(t-1)
#
# The priors: 
#    - uniform or Gaussian for the coefficients, 
#    - Inverse Gamma for the variance of noise in the state model

# "model_settings.m" - model setup
from typing import List
import numpy as np
# %%
# set the prior of the parameters: coefs, variance
# these coefficients shall be drawn from prior later. 
"""
    ssmPar = {"prior_type": List(int), 0 - Gaussian, 1 - Uniform
    }
"""
class ssmParBase:
    def __init__(self, prior_type: int, params: List(float)):
        """
            prior_type = 0 - Gaussian
                params[0]: mu
                params[1]: sigma
            prior_type = 1 - uniform
                params[0]: lower bound
                params[1]: upper bound
        """
        self.prior_type = prior_type


        if self.prior_type == 0:
            self.params = params
            self.theta = self.params[0] + np.random.randn()*self.params[1]
            # theta True = params[0] = mu
        elif self.prior_type == 1:
            self.params = params
            self.theta = np.random.uniform(self.params[0], self.params[1])
            # theta True = (params[1] + params[0])/2
        else:
            raise ValueError(f"Prior type cannot be found, got {prior_type}")
        # TODO: add self.coefs

class ssmPar(ssmParBase):
    self.coefs
    self.sigmaV
    self.sigmaW

# %%

p = [1,-thetaTrue(1:2)']; 
fprintf('Roots of 1-a1*z-a2*z^2 (should be outside of unit circle) \n'); 
root = roots(fliplr(p));   disp(root); 

alpha  = 2; beta= 1;       sigma2 = 1/gamrnd(alpha,1/beta);   
ssmPar.priorsVa = alpha;   ssmPar.priorsVb = beta;  % Inverse Gamma for sV
ssmPar.sigmaV   = 1+ 0*sqrt(sigma2);  
fprintf('True std of the model noise: %2.1f \n', ssmPar.sigmaV ); 

%% Settings of the state space model 
p=2; q=1;   
fnt =@(n)0*cos(1.2*(n-1) ); 

ssmPar.fnt    = fnt;   
ssmPar.sigmaW = .5;     % std of the observation noise
ssmPar.sigmaV = 1;      % std of the noise in the state model
ssmPar.p   = p; 
ssmPar.q   = q;

% % % Get Phi
meanFlag = 0; 
Terms    = @(XX,VV) terms_narma(XX,VV, p,q,meanFlag);    
Phi_n    = @(XX,VV,tn) Terms(XX,VV)*thetaTrue+fnt(tn);   % Phi_n dependes on time
ssmPar.Terms = Terms; 
ssmPar.Phi_n = Phi_n;

clearvars -except ssmPar; 