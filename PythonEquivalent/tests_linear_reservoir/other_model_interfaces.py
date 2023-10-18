# %%
import os
current_path = os.getcwd()
if current_path[-22:] != 'tests_linear_reservoir':
    os.chdir('tests_linear_reservoir')
    print("Current working directory changed to 'tests_linear_reservoir'.")   
import sys
sys.path.append('../') 

from model.model_interface import ModelInterface
from functions.utils import normalize_over_interval
import scipy.stats as ss
import numpy as np
import pandas as pd
from typing import Optional, Any

# %%
# Difference between Bulk case and Decimation case:
# Bulk case:
# - idea: only the estimation at the end of the interval matters, intermediate steps are not so important
# - input model: generated inputs sums to observed input; obs uncertainty only on observed bulk sum
# - transition model: totally based on observation
# Decimation case:
# - idea: find a model based on observed values at each time step and estimate from there
# - input model: generate inputs based on information on all input values
# - transition model: based on observation and transition steps

# %%
class ModelInterfaceBulk(ModelInterface):
    def get_bulk_obs_std(self) -> float:
        """Get bulk observation uncertainty

        Returns:
            float: bulk observation uncertainty
        """
        obs_interval = self.observed_ind[-1] - self.observed_ind[-2]
        input_std = self.influx[self.observed_ind].std(ddof=1)
        adjusted_std = np.sqrt(obs_interval) * input_std
        # adjusted_std = input_std
        return adjusted_std
    
    def input_model(self, start_ind: int, end_ind: int) -> None:
        """Input model for linear reservoir


        Ut: influx at time t
        R't = N(Ut, Ut_std)

        Args:
            start_ind (int): start index of the input time series
            end_ind (int): end index of the input time series

        Returns:
            np.ndarray: Rt
        """
        sig_u = self.get_bulk_obs_std()
        
        R_prime = np.zeros((self.N, end_ind - start_ind))
        U = self.influx[start_ind:end_ind].values
        sig_r = self.theta.input_model/5.

        for n in range(self.N):
            R_prime[n,:] = ss.norm(U, scale=sig_u).rvs()
            if R_prime[n,:][R_prime[n,:] >0].size == 0:
                R_prime[n,:][R_prime[n,:] <= 0] = 10**(-8)
            else:
                R_prime[n,:][R_prime[n,:] <= 0] = min(10**(-8), min(R_prime[n,:][R_prime[n,:] > 0]))
            R_prime[n,:] = normalize_over_interval(R_prime[n,:], U[0] + ss.norm(0, sig_r).rvs())

        return R_prime
    
    def transition_model_probability(self, X_1toT: np.ndarray) -> np.ndarray:
        return 1.
    

# %%
class ModelInterfaceDeci(ModelInterface):
    def __init__(self, df: pd.DataFrame, customized_model: Any | None = None, theta_init: dict[str, Any] | None = None, config: dict[str, Any] | None = None, num_input_scenarios: int | None = 10) -> None:
        super().__init__(df, customized_model, theta_init, config, num_input_scenarios)

        input_obs_ind = np.arange(self.T)[self.df['is_obs']]
        # data = self.influx.to_numpy()
        # self.interpolated = np.interp(np.arange(self.T), input_obs_ind, data[input_obs_ind])
        data = self.influx.to_numpy()
        mean = data[input_obs_ind].mean()
        std = data[input_obs_ind].std(ddof=1)
        data[~input_obs_ind] = ss.norm(mean, std).rvs(data[~input_obs_ind].size)
        self.interpolated = data

    def input_model(self, start_ind: int, end_ind: int) -> None:
        """Input model for linear reservoir


        Ut: influx at time t
        R't = N(Ut, Ut_std)

        Args:
            start_ind (int): start index of the input time series
            end_ind (int): end index of the input time series

        Returns:
            np.ndarray: Rt
        """

            
        R_prime = np.zeros((self.N, end_ind - start_ind))
        sig_r = self.theta.input_model
        R_hat = self.interpolated[start_ind:end_ind]

        for n in range(self.N):
            R_prime[n,:] = ss.norm(R_hat, scale=sig_r).rvs()
            if R_prime[n,:][R_prime[n,:] >0].size == 0:
                R_prime[n,:][R_prime[n,:] <= 0] = 10**(-8)
            else:
                R_prime[n,:][R_prime[n,:] <= 0] = min(10**(-8), min(R_prime[n,:][R_prime[n,:] > 0]))

        return R_prime

    
# %%
