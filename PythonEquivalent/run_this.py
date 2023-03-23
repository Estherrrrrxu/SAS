# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from functions.estimator import SSModel

# %%
# get a chopped dataframe
df = pd.read_csv("Dataset.csv", index_col= 0)
T = 50
interval = 1
df = df[:T:interval]
# %%
default_model = SSModel(data_df=df)
# %%
# CHECK sMC
default_model.N = 10
# set default theta
theta_init = {
    'to_estimate': {'k':{"prior_dis": "normal", "prior_params":[1.2,0.3], 
                            "update_dis": "normal", "update_params":[0.05]},
                    'output_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.0005], 
                            "update_dis": "normal", "update_params":[0.00005]}},
    'not_to_estimate': {'input_uncertainty': 0.254*1./24/60*15}
}
# process theta
default_model._theta_init = theta_init
default_model._process_theta()
X,A,W,R = default_model.run_sMC([1.,0.00005])
from functions.utils import plot_MLE
plot_MLE(X,A,W,R,default_model.K,df,default_model.influx, default_model.outflux,left = 0, right = default_model.K)
# %%
# AA = np.broadcast_to(A, (default_model.D, *A.shape))
# WW = np.zeros((default_model.D,default_model.N))
# XX = np.zeros((default_model.D,default_model.N,default_model.K+1))
# RR = np.zeros((default_model.D,default_model.N,default_model.K))
# default_model.run_pMCMC([1.,0.00005], X[np.newaxis,:,:],A[np.newaxis,:,:],W[np.newaxis,:],R[np.newaxis,:,:])




# %%

default_model.run_pGS_SAEM(num_particles=10,num_params=5, len_MCMC=15)
# %%
plt.figure()
plt.subplot(2,1,1)
plt.plot(default_model.theta_record[:,0])
plt.ylabel(r"$\theta$")
plt.xlabel("MCMC Chain")
plt.subplot(2,1,2)
plt.plot(default_model.theta_record[:,1])
plt.ylabel(r"$\theta$")
plt.xlabel("MCMC Chain")
# plt.plot([0,chain_len],[theta_record.T[10:].mean(),theta_record.T[10:].mean()],'r',label = "prior")
plt.legend()

# %%
