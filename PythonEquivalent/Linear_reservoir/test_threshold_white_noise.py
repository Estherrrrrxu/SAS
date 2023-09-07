# %%
import os
current_path = os.getcwd()
if current_path[-16:] != "Linear_reservoir":
    os.chdir("Linear_reservoir")
    print("Current working directory changed to 'Linear_reservoir'.")   
import sys
sys.path.append('../') 
from model.model_interface import ModelInterface, Parameter
from model.ssm_model import SSModel
from model.your_model import LinearReservoir
import pandas as pd

from model.utils_chain import Chain
from functions.utils import plot_MLE, plot_scenarios, normalize_over_interval
import matplotlib.pyplot as plt
from Linear_reservoir.data_threshold_white_noise import *
from copy import deepcopy
import scipy.stats as ss

# %%
class ModelInterfaceWN(ModelInterface):
    def update_model(
            self,
            theta_new: Optional[List[float]] = None
        ) -> None:       
        if theta_new is None:
            theta_new = []
            for key in self._theta_to_estimate:
                theta_new.append(self.prior_model[key].rvs())

        for i, key in enumerate(self._theta_to_estimate):
            self._theta_init['to_estimate'][key]['current_value'] = theta_new[i]
        # transition model param [0] is k to estimate, and [1] is fixed dt
        transition_param = [self._theta_init['to_estimate']['k']['current_value'], self.config['dt']]

        # observation uncertainty param is to estimate
        obs_param = self._theta_init['to_estimate']['obs_uncertainty']['current_value']

        # input uncertainty param is to estimate
        input_param = [self._theta_init['to_estimate']['input_mean']['current_value'], 
                       self._theta_init['to_estimate']['input_std']['current_value'],
                       self._theta_init['to_estimate']['input_uncertainty']['current_value']
                       ]

        # initial state param is to estimate
        init_state = self._theta_init['to_estimate']['initial_state']['current_value']

        self.theta = Parameter(
                            input_model=input_param,
                            transition_model=transition_param, 
                            observation_model=obs_param,
                            initial_state=init_state
                            )
        return 
    def input_model(
            self
        ) -> None:
        """Input model for linear reservoir

        Rt' ~ N(Ut, sigma_ipt)
        """
        for n in range(self.N):
            self.R[n,:] = ss.norm(loc=self.theta.input_model[0], scale=self.theta.input_model[1]).rvs()
            self.R[n,:][self.R[n,:] < 0] = 0     
            normalized = normalize_over_interval(self.R[n,:], self.observed_ind, self.influx)
            self.R[n,:] = normalized + ss.norm(loc=0, scale=self.theta.input_model[2]).rvs(self.T)
            self.R[n,:][self.R[n,:] < 0] = 0


        return 


#%%

def run_given_case(case: Cases, unified_color: Optional[bool] = False) -> None:
    # %%
    case = weekly_bulk_true_q
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
                                            "prior_params":[df_obs['Q_obs'].iloc[0], 0.0005],
                                            "search_dis": "normal", "search_params":[0.0005],
                                            "is_nonnegative": True
                            },
                        'obs_uncertainty':{"prior_dis": "uniform", 
                                            "prior_params":[0.000008,0.00006], 
                                            "search_dis": "normal", "search_params":[0.000001],
                                            "is_nonnegative": True
                            },
                        'input_uncertainty':{"prior_dis": "normal", 
                                                "prior_params":[0.002,0.002],
                                                "search_dis": "normal", "search_params":[0.001],
                                                "is_nonnegative": True
                            },
                        'input_mean':{"prior_dis": "normal",
                                            "prior_params":[-0.01, 0.001],
                                            "search_dis": "normal", "search_params":[0.01],
                                            "is_nonnegative": False
                            },
                        'input_std':{"prior_dis": "normal",
                                            "prior_params":[0.02,0.005], 
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
        num_input_scenarios = 15,
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
        num_parameter_samples = 20,
        len_parameter_MCMC = 35,
        learning_step = 0.75,
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


    
    fig, ax = plot_scenarios(df, df_obs, model, 1, unified_color)
    fig.suptitle(f"{case_name}")
    fig.show()
    fig, ax = plot_scenarios(df, df_obs, model, 30, unified_color)
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
