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
