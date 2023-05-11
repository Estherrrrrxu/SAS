# %%
from model.model_interface import ModelInterface
from model.ssm_model import SSModel
import pandas as pd
from model.utils_chain import Chain
from functions.utils import plot_MLE, plot_base
import matplotlib.pyplot as plt


# %%
df = pd.read_csv("Data/linear_reservoir.csv", index_col= 0)
# st, et = 20, 100
st, et = 300, 400
interval = 1
df = df[st:et]
df_obs = df[::interval]
plot_base(df, df_obs)
df_obs.index = range(len(df_obs))
# %%
# define theta_init
theta_init = {
    'to_estimate': {'k':{"prior_dis": "normal", "prior_params":[1.5,0.03], 
                            "search_dis": "normal", "search_params":[0.05]
                        },
                    'obs_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.0001], 
                            "search_dis": "normal", "search_params":[0.00001],
                        }
                    },
    'not_to_estimate': {'input_uncertainty': 0.254*1./24/60*15}
}

config = {'dt': interval*1./24/60*15}

# initialize model interface settings
model_interface = ModelInterface(
    df = df_obs,
    customized_model = None,
    theta_init = theta_init,
    config = config,
    num_input_scenarios = 5
)
# %%
chain = Chain(
    model_interface = model_interface,
    theta=[1., 0.00005]
)
chain.run_sequential_monte_carlo()
plot_MLE(chain.state,df,df_obs)
# %%
chain.run_particle_MCMC()
plot_MLE(chain.state,df,df_obs)



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
ax[0].set_xticks([])
ax[1].set_ylabel(r"$\theta_{obs}$")
ax[1].set_xlabel("MCMC Chain length")


# %%
plot_MLE(model.state,df,df_obs)

# %%
