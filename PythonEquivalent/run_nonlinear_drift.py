# %%
import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from functions.utils import plot_MLE
from functions.link import ModelLink
from functions.estimator import SSModel
import scipy.stats as ss
# %%
df = pd.read_csv("Data/Nonlinear_Drift.csv")
config = {
        'dt': 0.01,
        'influx': None,
        'outflux': 'Y'
    }

theta_init = {
            'to_estimate': {'1':{"prior_dis": "normal", "prior_params":[0,0.3], 
                                    "update_dis": "normal", "update_params":[0.05]
                                },
                            'obs_uncertainty':{"prior_dis": "uniform", "prior_params":[0.1,0.5], 
                                    "update_dis": "normal", "update_params":[0.05],
                                }
                            },
            'not_to_estimate': {}
        }





model_link = ModelLink(
    df=df, 
    num_input_scenarios=15, 
    config=config, 
    theta_init=theta_init
    )
default_model = SSModel(model_link)

# %%
# state = default_model.run_sequential_monte_carlo([1.,0.00005])
# plot_MLE(state,df,left = 0, right = 50)

# state = default_model.run_particle_MCMC(state,theta = [1.,0.00005])
# plot_MLE(state,df,left = 0, right = 50)
# %%
default_model.run_particle_Gibbs_SAEM(
    num_parameter_samples=20,
    len_MCMC=45)
# %%
fig, ax = plt.subplots(2,1,figsize=(10,5))
ax[0].plot(default_model.theta_record[:,0])
ax[1].plot(default_model.theta_record[:,1])
ax[0].set_ylabel(r"$k$")
ax[1].set_ylabel(r"$\theta_{obs}$")
ax[1].set_xlabel("MCMC Chain length")
# %%
