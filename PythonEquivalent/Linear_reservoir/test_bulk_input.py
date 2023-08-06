# %%
import os
current_path = os.getcwd()
if current_path[-16:] != "Linear_reservoir":
    os.chdir("Linear_reservoir")
    print("Current working directory changed to 'Linear_reservoir'.")   
import sys
sys.path.append('../') 
from model.model_interface import ModelInterface
from model.ssm_model import SSModel
import pandas as pd

from model.utils_chain import Chain
from functions.utils import plot_MLE, plot_scenarios
import matplotlib.pyplot as plt
from Linear_reservoir.input_data_generation import weekly_bulk, biweekly_bulk, weekly_bulk_true_q

# %%
case = weekly_bulk

df = case.df
df_obs = case.df_obs
obs_made = case.obs_made
case_name = case.case_name

# %%
# define theta_init
theta_init = {
    'to_estimate': {'k':{"prior_dis": "normal", "prior_params":[1.5,0.3], 
                            "search_dis": "normal", "search_params":[0.05]
                        },
                    'obs_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.01], 
                            "search_dis": "normal", "search_params":[0.00001],
                        }
                    },
    'not_to_estimate': {'input_uncertainty': 0.254*1./24/60*15, 'state_peak': 0.0005}
}

config = {'observed_made_each_step': obs_made}

# initialize model interface settings
model_interface = ModelInterface(
    df = df_obs,
    customized_model = None,
    theta_init = theta_init,
    config = config,
    num_input_scenarios = 5
)

# %%
try: 
    chain = Chain(
        model_interface = model_interface,
        theta=[1., 0.000005]
    )
    chain.run_sequential_monte_carlo()
    plot_MLE(chain.state,df,df_obs,chain.pre_ind,chain.post_ind)

    chain.run_particle_MCMC()
    plot_MLE(chain.state,df,df_obs,chain.pre_ind,chain.post_ind)
except IndexError:
    print("Index needs to change for Chain obj!") 


# %%
# run PMCMC
model = SSModel(
    model_interface = model_interface,
    num_parameter_samples = 10,
    len_parameter_MCMC = 5,
    learning_step = 0.75
)
model.run_particle_Gibbs_AS_SAEM()
# %%
fig, ax = plt.subplots(2,1,figsize=(10,5))
ax[0].plot(model.theta_record[:,0])
ax[0].plot([0,15],[1,1],'r:',label="true value")
ax[1].plot(model.theta_record[:,1])
ax[1].plot([0,15],[0.00005,0.00005],'r:',label="true value")
ax[0].set_ylabel(r"$k$")
ax[0].set_xticks([])
ax[1].set_ylabel(r"$\theta_{v}$")
ax[1].set_xlabel("MCMC Chain length")
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)
fig.suptitle(f"Parameter estimation for {case_name}")

# %%
fig, ax = plot_scenarios(df, df_obs, model, 10)
fig.suptitle(f"{case_name}")

# %%
