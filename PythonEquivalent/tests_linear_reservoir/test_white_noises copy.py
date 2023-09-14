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
from functions.utils import plot_MLE, plot_scenarios, normalize_over_interval
from functions.get_dataset import get_different_input_scenarios

import matplotlib.pyplot as plt
import scipy.stats as ss
from typing import Optional, List
import pandas as pd

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
        obs_param = self._theta_init['not_to_estimate']['obs_uncertainty']

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
    
    # def input_model(
    #         self
    #     ) -> None:
    #     """Input model for linear reservoir

    #     Rt' ~ N(Ut, sigma_ipt)
    #     """
    #     for n in range(self.N):
    #         self.R[n,:] = ss.norm(loc=self.influx, scale=self.theta.input_model[2]).rvs()
    #         self.R[n,:][self.R[n,:] < 0] = 0 
    #     return 
    

    def input_model(
            self
        ) -> None:
        """Input model for linear reservoir

        Rt' ~ N(Ut, sigma_ipt)
        """
        for n in range(self.N):
            self.R[n,:] = ss.norm(loc=self.theta.input_model[0], scale=self.theta.input_model[1]).rvs()
            if self.R[n,:][self.R[n,:] >0].size == 0:
                self.R[n,:][self.R[n,:] <= 0] = 10**(-8)
            else:
                self.R[n,:][self.R[n,:] <= 0] = min(10**(-8), min(self.R[n,:][self.R[n,:] > 0]))     
            normalized = normalize_over_interval(self.R[n,:], self.observed_ind, self.influx)
            self.R[n,:] = normalized + ss.norm(loc=0, scale=self.theta.input_model[2]).rvs(self.T)
            self.R[n,:][self.R[n,:] < 0] = 0
        return 


# %%
def plot_parameters_linear_reservoir(
        len_parameter_MCMC: int,
        k: List[float],
        S0: List[float],
        theta_u: List[float],
        u_mean: List[float],
        u_std: List[float],
        S0_true: float,
        case_name: str
    ) -> None:
    fig, ax = plt.subplots(5,1,figsize=(10,5))
    ax[0].plot(k)
    ax[1].plot(S0)
    ax[2].plot(theta_u)
    ax[3].plot(u_mean)
    ax[4].plot(u_std)


    ax[0].plot([0,len_parameter_MCMC],[1,1],'r:',label="true value")
    ax[1].plot([0,len_parameter_MCMC],[S0_true, S0_true],'r:',label="true approximation")
    ax[2].plot([0,len_parameter_MCMC],[0.00005,0.00005],'r:',label="true value")
    ax[3].plot([0,len_parameter_MCMC],[-0.01,-0.01],'r:',label="true value")
    ax[4].plot([0,len_parameter_MCMC],[0.02,0.02],'r:',label="true value")

    ax[0].set_ylabel(r"$k$")
    ax[0].set_xticks([])
    ax[1].set_ylabel(r"$S_0$")
    ax[1].set_xticks([])
    ax[2].set_ylabel(r"$\theta_{v}$")
    ax[2].set_xticks([])
    ax[3].set_ylabel(r"$\mu_u$")
    ax[3].set_xticks([])
    ax[4].set_ylabel(r"$\sigma_u$")
    ax[4].set_xlabel("MCMC iteration")
    
    ax[0].legend(frameon=False)
    ax[1].legend(frameon=False)
    ax[2].legend(frameon=False)
    ax[3].legend(frameon=False)
    ax[4].legend(frameon=False)

    fig.suptitle(f"Parameter estimation for {case_name}")
    fig.show()

# %%
def run_linear_reservoir(
        case, 
        theta_init: Optional[dict],
        unified_color: Optional[bool] = False,
        num_input_scenarios: Optional[int] = 5,
        plot_preliminary: Optional[bool] = False,
        num_parameter_samples: Optional[int] = 10,
        len_parameter_MCMC: Optional[int] = 15,
        learning_step: Optional[int] = 0.6,
        start_ind: Optional[int] = 1
    ) -> None:

    # Get data
    df = case.df
    df_obs = case.df_obs
    obs_made = case.obs_made
    case_name = case.case_name

    config = {'observed_made_each_step':obs_made}

    # initialize model interface settings
    model_interface = ModelInterfaceWN(
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

        chain.run_particle_MCMC()
        plot_MLE(chain.state,df,df_obs,chain.pre_ind,chain.post_ind)

    # 
    # run PMCMC
    model = SSModel(
        model_interface = model_interface,
        num_parameter_samples = num_parameter_samples,
        len_parameter_MCMC = len_parameter_MCMC,
        learning_step = learning_step,
    )
    model.run_particle_Gibbs_AS_SAEM()
    #
    # plot parameters
    plot_parameters_linear_reservoir(
        len_parameter_MCMC,
        model.theta_record[:,0],
        model.theta_record[:,1],
        model.theta_record[:,2],
        model.theta_record[:,3],
        model.theta_record[:,4],
        df_obs['Q_obs'].iloc[0],
        case_name
    )
    # plot scenarios
    fig, ax = plot_scenarios(df, df_obs, model, start_ind, unified_color)
    fig.suptitle(f"{case_name}")
    fig.show()
    fig, ax = plot_scenarios(df, df_obs, model, 20, unified_color)
    fig.suptitle(f"{case_name}")
    fig.show()
    return

# %%
if __name__ == "__main__":

    num_input_scenarios = 15
    num_parameter_samples = 20
    len_parameter_MCMC = 25
    plot_preliminary = True
    learning_step = 0.6
    start_ind = 0
    unified_color = True
    perfects = []

    df = pd.read_csv(f"../Data/WhiteNoise/stn_5_30.csv", index_col= 0)
    interval = [0,100]

    perfect, instant_gaps_2_d, instant_gaps_5_d, weekly_bulk, biweekly_bulk, weekly_bulk_true_q = get_different_input_scenarios(df, interval, plot=False)
    #%%
    case = perfect
        # Get data
    df = case.df
    df_obs = case.df_obs
    obs_made = case.obs_made
    case_name = case.case_name
    theta_init = {
        'to_estimate': {'k':{"prior_dis": "normal", 
                                "prior_params":[1.,0.001], 
                                "search_dis": "normal", "search_params":[0.001],
                                "is_nonnegative": True
                            },
                        'initial_state':{"prior_dis": "normal", 
                                            "prior_params":[df_obs['Q_obs'][0], 0.001],
                                            "search_dis": "normal", "search_params":[0.001],
                                            "is_nonnegative": True
                            },
                        'input_mean':{"prior_dis": "normal",
                                        "prior_params":[df_obs['J_obs'].mean(), 0.001],
                                        "search_dis": "normal", "search_params":[0.001],
                                        "is_nonnegative": True
                            },
                        'input_std':{"prior_dis": "normal",
                                        "prior_params":[df_obs['J_obs'].std(ddof=1), 0.001],
                                        "search_dis": "normal", "search_params":[0.001],
                                        "is_nonnegative": True
                            },
                        'input_uncertainty':{"prior_dis": "normal",
                                                "prior_params":[df_obs['J_obs'].std(ddof=1)/5, 0.001],
                                                "search_dis": "normal", "search_params":[0.001],
                                                "is_nonnegative": True
                            },
                        },
        'not_to_estimate': {'obs_uncertainty':0.00001}
    }
    (df['Q_obs'] - df['Q_true']).std(ddof=1)


    # for perfect in perfects:

    # run different cases
    run_linear_reservoir(perfect, theta_init, unified_color, num_input_scenarios, plot_preliminary, num_parameter_samples, len_parameter_MCMC, learning_step, start_ind)



    #%%
    # # %%
    # stn_ipts = [1, 2, 3, 4, 5]
    # stn_ratios = [0.5, 1, 2, 3, 4, 5, 6]
    #%%
    # perfects = []
    # for stn_ipt in stn_ipts:
    #     for stn_ratio in stn_ratios:
    #         stn_obs = stn_ipt * stn_ratio
    #         df = pd.read_csv(f"../Data/WhiteNoise/stn_{stn_ipt}_{stn_obs}.csv", index_col= 0)
    #         interval = [0,100]

    #         perfect, instant_gaps_2_d, instant_gaps_5_d, weekly_bulk, biweekly_bulk, weekly_bulk_true_q = get_different_input_scenarios(df, interval, plot=False)

    #         perfects.append(perfect)

    # # for perfect in perfects:

    # # run different cases
    # run_linear_reservoir(perfect, theta_init, unified_color, num_input_scenarios, plot_preliminary, num_parameter_samples, len_parameter_MCMC, learning_step, start_ind)




# %%
