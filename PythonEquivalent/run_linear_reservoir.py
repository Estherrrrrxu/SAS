# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from functions.utils import plot_MLE
from functions.link import ModelLink
from functions.estimator import SSModel
from functions.model_structure import LinearReservior
# %%
# get a chopped dataframe
df = pd.read_csv("Data/linear_reservoir.csv", index_col= 0)
T = 50
interval = 1
df = df[:T:interval]
# %%
model_link = ModelLink(
                        df=df, 
                        customized_model=LinearReservior,
                        num_input_scenarios=15
                        )
model_link._parse_config()
model_link._parse_theta_init()
model_link._update_model([1,0.00005])
model_link.input_generation()
print(model_link.R)
# %%
print(default_model = SSModel(ModelLink))

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
