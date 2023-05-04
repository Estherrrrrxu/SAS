# %%
from model.model_interface import ModelInterface
from model.ssm_model import SSModel
import pandas as pd
from model.utils_chain import Chain
from functions.utils import plot_MLE
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("Data/linear_reservoir.csv", index_col= 0)
T = 50
interval = 1
df = df[:T:interval]
# %%
# initialize model interface settings
model_interface = ModelInterface(
    df = df,
    customized_model = None,
    theta_init = None,
    config = None,
    num_input_scenarios = 5
)
# %%
# chain = Chain(
#     model_interface = model_interface,
#     theta=[1, 0.00005]
# )
# chain.run_sequential_monte_carlo()
# plot_MLE(chain.state,df,left = 0, right = 50)
# chain.run_particle_MCMC()
# plot_MLE(chain.state,df,left = 0, right = 50)


# %%
# run PMCMC
model = SSModel(
    model_interface = model_interface,
    num_parameter_samples = 10,
    len_parameter_MCMC = 15,
    learning_step = 0.75
)
model.run_particle_Gibbs_AS_SAEM()
# %%
fig, ax = plt.subplots(2,1,figsize=(10,5))
ax[0].plot(model.theta_record[:,0])
ax[1].plot(model.theta_record[:,1])
ax[0].set_ylabel(r"$k$")
ax[1].set_ylabel(r"$\theta_{obs}$")
ax[1].set_xlabel("MCMC Chain length")

# %%