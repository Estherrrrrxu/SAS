#  Estimate parameter by Particle Gibbs with Ancestor Sampling (PGAS) algorithm,
#  References:
#   [1] F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for
#   Particle Gibbs", Proceedings of the 2012 Conference on Neural
#   Information Processing Systems (NIPS), Lake Taho, USA, 2012.
#  and
#   [2] C. Andrieu, A. Doucet and R. Holenstein, "Particle Markov chain Monte
#   Carlo methods" Journal of the Royal Statistical Society: Series B,
#   2010, 72, 269-342.
#
# The script generates a batch of data y_{1:T} from a nonlinear time series model,
#
#   x_t = x_{t-1} + \Delta t * [J(t) - Q(t) * \Omega_Q(x_{t-1}) - d x_t / d T)],
#   y_t = 
#
#   x_t = S_T at t
#   y_t = \Delta C_Q(t) = C_Q(t) - C_Q(t-1)
#
# with v_t ~ N(0,sV) and w_t ~ N(0,sW). 
# We assume that sV, sW are given, but the parameters 
#           theta = {\Omega_Q(x_{t-1})}
# are unknown and are to be estimated.  
# The PGAS algorithm generates a Markov chain to sample the posterior  
# of the parameter and state: p(theta, x_{1:T} | y_{1:T}). 

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%
# read file
data_true = pd.read_csv("synthetic.csv",index_col=0)
# %%
ssmPar_thetaV = 'theta' 

# number of time steps in state space model
tN      = 1000
# uniform prior   
type    = 'evenObs_unif_tN'+str(tN)
ssmPar_prior = 2  
# Gaussian prior
# type   = 'Odd_GaussNp20'
# ssmPar_prior  = 1

sampletype= type+'_Np20nMC10k'
Obsdatafile    = 'Obsdata_'+type+'.mat'
figname        = sampletype;      
sampleFilename = 'sampledata_'+sampletype+'.mat'









