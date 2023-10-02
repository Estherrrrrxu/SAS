# %%
import os
current_path = os.getcwd()
if current_path[-22:] != 'tests_linear_reservoir':
    os.chdir('tests_linear_reservoir')
    print("Current working directory changed to 'tests_linear_reservoir'.")   
import sys
sys.path.append('../') 

from model.model_interface import ModelInterface, Parameter
from model.ssm_model import SSModel
from model.your_model import LinearReservoir

from model.utils_chain import Chain
from functions.utils import plot_MLE, plot_scenarios
from functions.get_dataset import get_different_input_scenarios

import matplotlib.pyplot as plt
import scipy.stats as ss
from typing import Optional, List
import pandas as pd
import numpy as np


# %%
def plot_parameters_linear_reservoir(
        len_parameter_MCMC: int,
        k: List[float],
        S0: List[float],
        theta_u: List[float],
        u_mean: List[float],
        S0_true: float,
        case_name: str
    ) -> None:
    fig, ax = plt.subplots(5,1,figsize=(10,5))
    ax[0].plot(k)
    ax[1].plot(S0)
    ax[2].plot(theta_u)
    ax[3].plot(u_mean)


    ax[0].plot([0,len_parameter_MCMC],[1,1],'r:',label="true value")
    ax[1].plot([0,len_parameter_MCMC],[S0_true, S0_true],'r:',label="true approximation")
    ax[2].plot([0,len_parameter_MCMC],[0.00005,0.00005],'r:',label="true value")
    ax[3].plot([0,len_parameter_MCMC],[-0.01,-0.01],'r:',label="true value")

    ax[0].set_ylabel(r"$k$")
    ax[0].set_xticks([])
    ax[1].set_ylabel(r"$S_0$")
    ax[1].set_xticks([])
    ax[2].set_ylabel(r"$\theta_{v}$")
    ax[2].set_xticks([])
    ax[3].set_ylabel(r"$\mu_u$")
    ax[3].set_xticks([])

    
    ax[0].legend(frameon=False)
    ax[1].legend(frameon=False)
    ax[2].legend(frameon=False)
    ax[3].legend(frameon=False)

    fig.suptitle(f"Parameter estimation for {case_name}")
    fig.show()

# %%
num_input_scenarios = 5
num_parameter_samples = 5
len_parameter_MCMC = 50
plot_preliminary = True
fast_convergence_phase_length = 15
start_ind = 0
unified_color = True
perfects = []

df = pd.read_csv(f"../Data/WhiteNoise/stn_5_30.csv", index_col= 0)
# df = pd.read_csv(f"../Data/RealPrecip/stn_5_30.csv", index_col= 0)
interval = [0,20]

perfect, instant_gaps_2_d, instant_gaps_5_d, weekly_bulk, biweekly_bulk, weekly_bulk_true_q = get_different_input_scenarios(df, interval, plot=False)

case = perfect
# Get data
df = case.df
df_obs = case.df_obs
obs_made = case.obs_made
case_name = case.case_name

config = {'observed_made_each_step':obs_made,
          'outflux': 'Q_true'}


# %%
print("The input mean: ", df_obs['J_obs'].mean())
print("The input std: ", df_obs['J_obs'].std())
print("The output std: ", df_obs['Q_obs'].std())
print("Initial state: ", df_obs['Q_true'].iloc[0]) 

k_true = 1.0
initial_state_true = df_obs['Q_true'].iloc[0]
input_uncertainty_true = df_obs['J_obs'].std(ddof=1)/5.0
obs_uncertainty_true = df_obs['Q_obs'].std(ddof=1)

# %%
theta_init = {
    "to_estimate": {
        "k": {
            "prior_dis": "normal",
            "prior_params": [1.2, 0.03],
            "is_nonnegative": True,
        },
        "initial_state": {
            "prior_dis": "normal",
            "prior_params": [df_obs['Q_obs'][0], 0.01],
            "is_nonnegative": True,
        },
        "input_uncertainty": {
            "prior_dis": "normal",
            "prior_params": [input_uncertainty_true, input_uncertainty_true/10.,
            ],
            "is_nonnegative": True,
        },
        "obs_uncertainty": {
            "prior_dis": "normal",
            "prior_params": [obs_uncertainty_true, obs_uncertainty_true/10.,
            ],
            "is_nonnegative": True,
        },
    },
    "not_to_estimate": {},
}

# %%
# initialize model interface settings
model_interface = ModelInterface(
    df = df_obs,
    customized_model = LinearReservoir,
    theta_init = theta_init,
    num_input_scenarios = num_input_scenarios,
    config = config
)

if plot_preliminary:
    chain = Chain(
        model_interface = model_interface
    )
    chain.run_sequential_monte_carlo()
    plot_MLE(chain.state,df,df_obs,chain.pre_ind,chain.post_ind)
    plt.plot(chain.state.X.T, ".")

    chain.run_particle_MCMC_AS()
    plot_MLE(chain.state,df,df_obs,chain.pre_ind,chain.post_ind)
    plt.plot(chain.state.X.T, ".")


    #%%

    # run PMCMC
    model = SSModel(
        model_interface=model_interface,
        num_parameter_samples=num_parameter_samples,
        len_parameter_MCMC=len_parameter_MCMC,
        fast_convergence_phase_length=fast_convergence_phase_length
    )
    model.run_particle_Gibbs_SAEM()
    # %%
    # plot parameters
    plot_parameters_linear_reservoir(
        len_parameter_MCMC,
        model.theta_record[:,0],
        model.theta_record[:,1],
        model.theta_record[:,2],
        model.theta_record[:,3],       
        df_obs['Q_obs'].iloc[0],
        case_name
    )
    #%%
k = model.theta_record[:,0]
initial_state = model.theta_record[:,1]
input_uncertainty = model.theta_record[:,2]
obs_uncertainty = model.theta_record[:,3]


    #%%

    fig, ax = plt.subplots(1,4,figsize=(10,5))
    ax[0].hist(k)
    ax[1].hist(initial_state)
    ax[2].hist(input_uncertainty)
    ax[3].hist(obs_uncertainty)


    ax[0].set_title(r"$k$")
    ax[1].set_title(r"$S_0$")
    ax[2].set_title(r"$\sigma_u$")
    ax[3].set_title(r"$\sigma_v$")

    ax[0].plot([k_true,k_true],[0,8],'r:',label="true value")
    ax[1].plot([initial_state_true,initial_state_true],[0,8],'r:',label="true approximation")
    ax[2].plot([input_uncertainty_true,input_uncertainty_true],[0,8],'r:',label="true value")
    ax[3].plot([obs_uncertainty_true,obs_uncertainty_true],[0,8],'r:',label="true value")

    ax[0].legend(frameon=False)
    ax[1].legend(frameon=False)
    ax[2].legend(frameon=False)
    ax[3].legend(frameon=False)

    fig.suptitle(f"Parameter estimation for {case_name}")
    fig.show()


    # %%
    # plot scenarios
    fig, ax = plot_scenarios(df, df_obs, model, start_ind, unified_color = False)
    fig.suptitle(f"{case_name}")
    fig.show()
    fig, ax = plot_scenarios(df, df_obs, model, 10, unified_color)
    fig.suptitle(f"{case_name}")
    fig.show()





# %%
from numba import vectorize
@vectorize(['float64(float64, float64, float64, float64)'])
def cal_KL(sigma1, sigma2, mu1, mu2):
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 1/2

# %%
e = df['J_true'].to_numpy() * model_interface.dt
sig_e = np.std(e, ddof=1)
phi = 1 - model.theta_record[:,0]* model_interface.dt
sig_q = np.sqrt(sig_e**2 / (1 - phi**2))

true_q_mean = df['Q_true'].to_numpy()
true_q_std = sig_q.mean()/30
obs_q_mean = model.output_record.mean(axis=0)
obs_q_std = model.output_record.std(axis=0,ddof=1)

KL = cal_KL(true_q_mean, obs_q_mean, true_q_std, obs_q_std)



    


# %%
