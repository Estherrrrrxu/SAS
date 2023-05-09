# %%
from model.model_interface import ModelInterface
from model.ssm_model import SSModel
import pandas as pd
from model.utils_chain import Chain
from functions.utils import plot_MLE
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("Data/linear_reservoir.csv", index_col= 0)
st = 0
et = 50
interval = 1
df = df[st:et:interval]

fig, ax = plt.subplots(2, 1, figsize=(8,5))
ax[0].bar(df.index, df['J_true'], 
          width = 1, color = 'b', alpha = 0.5, 
          label = 'Truth')
ax[0].plot(df['J_obs'], '*', color = 'b',label = 'Observation')
ax[0].invert_yaxis()
ax[0].set_ylabel("Precipitation [mm]")
ax[0].legend(frameon = False)
ax[0].set_xticks([])
ax[0].set_title("Preciptation")

ax[1].plot(df['Q_true'], color = 'b', alpha = 0.5, label = 'Truth')    
ax[1].plot(df['Q_obs'], '*', color = 'b', label = 'Observation')
ax[1].set_ylabel("Discharge [mm]")
ax[1].set_xlabel("Time [day]")
ax[1].legend(frameon = False)
ax[1].set_title("Discharge")


# %%
# define theta_init
theta_init = {
    'to_estimate': {'k':{"prior_dis": "normal", "prior_params":[1.5,0.3], 
                            "search_dis": "normal", "search_params":[0.05]
                        },
                    'obs_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.0001], 
                            "search_dis": "normal", "search_params":[0.00001],
                        }
                    },
    'not_to_estimate': {'input_uncertainty': 0.254*1./24/60*15}
}
# initialize model interface settings
model_interface = ModelInterface(
    df = df,
    customized_model = None,
    theta_init = theta_init,
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
