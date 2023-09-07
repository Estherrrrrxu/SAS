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
from Linear_reservoir.test_data_white_noise_filter import *

import scipy.stats as ss
# %%
class ModelInterfaceWN(ModelInterface):
            
    def input_model(
            self
        ) -> None:
        """Input model for linear reservoir

        Rt' ~ N(Ut, sigma_ipt)
        """
        for n in range(self.N):
            self.R[n,:] = ss.norm(loc=self.influx, scale=self.theta.input_model).rvs()
            self.R[n,:][self.R[n,:] < 0] = 0
        return 







def run_given_case(case: Cases, unified_color: Optional[bool] = False) -> None:
    df = case.df
    df_obs = case.df_obs
    obs_made = case.obs_made
    case_name = case.case_name

    theta_init = {
        'to_estimate': {'k':{"prior_dis": "normal", 
                                "prior_params":[1.2,0.1], 
                                "search_dis": "normal", "search_params":[0.1],
                                "is_nonnegative": True
                            },
                        'initial_state':{"prior_dis": "normal", 
                                            "prior_params":[df_obs['Q_obs'].iloc[0], 0.005],
                                            "search_dis": "normal", "search_params":[0.0005],
                                            "is_nonnegative": True
                            },
                        'obs_uncertainty':{"prior_dis": "uniform", 
                                            "prior_params":[0.00001,0.0001], 
                                            "search_dis": "normal", "search_params":[0.0001],
                                            "is_nonnegative": True
                            },
                        'input_uncertainty':{"prior_dis": "normal", 
                                                "prior_params":[0.005,0.002],
                                                "search_dis": "normal", "search_params":[0.001],
                                                "is_nonnegative": True
                            },
                        },
        'not_to_estimate': {}
    }

    config = {'observed_made_each_step':obs_made}

    # initialize model interface settings
    model_interface = ModelInterfaceWN(
        df = df_obs,
        customized_model = LinearReservoir,
        theta_init = theta_init,
        num_input_scenarios = 5,
        config = config
    )

    #
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


    # 
    # run PMCMC
    model = SSModel(
        model_interface = model_interface,
        num_parameter_samples = 10,
        len_parameter_MCMC = 15,
        learning_step = 0.6,
    )
    model.run_particle_Gibbs_AS_SAEM()
    # 
    fig, ax = plt.subplots(4,1,figsize=(10,5))
    ax[0].plot(model.theta_record[:,0])
    ax[0].plot([0,15],[1,1],'r:',label="true value")
    ax[1].plot(model.theta_record[:,1])
    ax[1].plot([0,15],[0.04, 0.04],'r:',label="true value")
    ax[2].plot(model.theta_record[:,2])
    ax[2].plot([0,15],[0.00006,0.00006],'r:',label="true value")
    ax[3].plot(model.theta_record[:,3])
    ax[3].plot([0,15],[0.002,0.002],'r:',label="true value")
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
    fig.show()


    # 
    fig, ax = plot_scenarios(df, df_obs, model, 10, unified_color)
    fig.suptitle(f"{case_name}")
    fig.show()

    return

# %%
if __name__ == "__main__":
    print("Running perfect case...")
    run_given_case(perfect, unified_color=True)
    #%%
    print("Running instant gap 2 days case...")
    run_given_case(instant_gaps_2_d, unified_color=True)
    print("Running instant gap 5 days case...")
    run_given_case(instant_gaps_5_d, unified_color=True)
    print("Running weekly bulk case...")
    run_given_case(weekly_bulk, unified_color=True)
    print("Running biweekly bulk case...")
    run_given_case(biweekly_bulk, unified_color=True)
    print("Running weekly bulk true Q case...")
    #%%
    run_given_case(weekly_bulk_true_q, unified_color=True)
    print("Done!")


# %%


# %%
