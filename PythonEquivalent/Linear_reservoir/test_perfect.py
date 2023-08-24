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
from model.your_model import LinearReservoir
import pandas as pd

from model.utils_chain import Chain
from functions.utils import plot_MLE, plot_scenarios
import matplotlib.pyplot as plt
from Linear_reservoir.test_data import perfect

# %%
case = perfect

df = case.df
df_obs = case.df_obs
obs_made = case.obs_made
case_name = case.case_name


# %%
theta_init = {
    'to_estimate': {'k':{"prior_dis": "normal", 
                            "prior_params":[1.5,0.03], 
                            "search_dis": "normal", "search_params":[0.1],
                            "is_nonnegative": True
                        },
                    'initial_state':{"prior_dis": "normal", 
                                        "prior_params":[df_obs['Q_obs'].iloc[0], 0.005],
                                        "search_dis": "normal", "search_params":[0.001],
                                        "is_nonnegative": True
                        },
                    'obs_uncertainty':{"prior_dis": "uniform", 
                                        "prior_params":[0.000001,0.0001], 
                                        "search_dis": "normal", "search_params":[0.0001],
                                        "is_nonnegative": True
                        },
                    'input_uncertainty':{"prior_dis": "uniform", 
                                            "prior_params":[0.0,0.005],
                                            "search_dis": "normal", "search_params":[0.001],
                                            "is_nonnegative": True
                        },
                    },
    'not_to_estimate': {}
}
# initialize model interface settings
model_interface = ModelInterface(
    df = df_obs,
    customized_model = LinearReservoir,
    theta_init = theta_init,
    num_input_scenarios = 5
)

# %%
try: 
    chain = Chain(
        model_interface = model_interface
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
    len_parameter_MCMC = 15,
    learning_step = 0.6
)
model.run_particle_Gibbs_AS_SAEM()
# %%
fig, ax = plt.subplots(4,1,figsize=(10,5))
ax[0].plot(model.theta_record[:,0])
ax[0].plot([0,15],[1,1],'r:',label="true value")
ax[1].plot(model.theta_record[:,1])
ax[1].plot([0,15],[df_obs['Q_true'].iloc[0],df_obs['Q_true'].iloc[0]],'r:',label="true value")
ax[2].plot(model.theta_record[:,2])
ax[2].plot([0,15],[0.00005,0.00005],'r:',label="true value")
ax[3].plot(model.theta_record[:,3])
ax[3].plot([0,15],[0.0005,0.0005],'r:',label="true value")
ax[0].set_ylabel(r"$k$")
ax[0].set_xticks([])
ax[1].set_ylabel(r"$S_0$")
ax[1].set_xticks([])
ax[2].set_ylabel(r"$\theta_{v}$")
ax[2].set_xticks([])
ax[3].set_ylabel(r"$\theta_{u}$")
ax[3].set_xlabel("MCMC Chain length")
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)
fig.suptitle(f"Parameter estimation for {case_name}")


# %%
fig, ax = plot_scenarios(df, df_obs, model, 10)
fig.suptitle(f"{case_name}")


# %%
