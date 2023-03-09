# %%
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from tqdm import tqdm
# %%
# ==================
# Get data
# ==================
df = pd.read_csv("Dataset.csv", index_col= 0)
T = 500
interval = 1
df = df[:T]
J_obs = df['J_obs'][::interval]
Q_obs = df['Q_obs'][::interval]
J = J_obs.values
Q = Q_obs.values
# true model inputs
k = 1
delta_t = 1./24/60*15
theta_ipt = 0.254*delta_t
theta_obs = 0.00005
K = len(J)
delta_t *= interval
# estimation inputs
sig_v = theta_ipt
sig_w = theta_obs

# %%
num_scenarios = 10
param_samples= 10
chain_len= 10
prior_mean = 0.9
prior_sd = 0.2
L = chain_len
D = param_samples
N = num_scenarios
# %%
theta_record = pd.read_csv(f"Results/theta_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",header = None)
input_record = pd.read_csv(f"Results/input_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",header = None)
WW = pd.read_csv(f"Results/W_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",header = None)

AA = pd.read_csv(f"Results/A_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",header = None)
XX= pd.read_csv(f"Results/X_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",header = None)

AA = AA.values.reshape(D+1, N, K+1)
XX = XX.values.reshape(D+1, N, K+1)

theta_record = theta_record.values
input_record = input_record.values
WW = WW.values

# %%
plt.figure()
plt.plot(theta_record)
plt.ylabel(r"$\theta$")
plt.xlabel("MCMC Chain")
plt.figure()
plt.hist(theta_record,bins = 10,label = "theta_distribution")
x = np.linspace(ss.norm(prior_mean,prior_sd).ppf(0.01),ss.norm(prior_mean,prior_sd).ppf(0.99), 100)
plt.plot(x,ss.norm(prior_mean,prior_sd).pdf(x),label = "prior")
plt.xlabel(r"$\theta$")
plt.legend()

# %%
id = np.argmax(np.nan_to_num(WW[:,-1],0))

left = 0
right = 500

B = np.zeros(K+1).astype(int)
B[-1] = AA[-2,:,-1][id]
for i in reversed(range(1,K+1)):
    B[i-1] =  AA[-2,:,i][B[i]]
MLE = np.zeros(K+1)
for i in range(K+1):
    MLE[i] = XX[-2,B[i],i]
# ------------
# ------------
plt.figure()
plt.subplot(2,1,1)
plt.plot(df['J_true'],label = "J true")
plt.plot(J_obs,"*",label = "J obs")
plt.plot(input_record.T,'r.',alpha = 0.01)
plt.legend(frameon = False)
plt.xlim([left,right])
plt.title('J')

plt.subplot(2,1,2)
plt.plot(Q_obs,".", label = "Observation")
plt.plot(df['Q_true'],'k', label = "Hidden truth")
plt.plot(np.linspace(0,T,len(MLE)-1),MLE[1:],'r:', label = "One Traj/MLE")
plt.legend(frameon = False)
plt.title(f"sig_v {round(sig_v,5)}, sig_w {sig_w}")
plt.xlim([left,right])
plt.tight_layout()
# %%
# ==================
# check particles
# ==================
for kk in range(10):
    plt.figure()
    plt.plot(MLE)
    # plt.plot(A[:,kk],X[:,kk],'r.')
# %%
