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
import matplotlib.pyplot as plt

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
    def __init__(self, df: pd.DataFrame, customized_model: Any | None = None, theta_init: dict[str, Any] | None = None, config: dict[str, Any] | None = None, num_input_scenarios: int | None = 10) -> None:
        # initialize the model interface
        super().__init__(df, customized_model, theta_init, config, num_input_scenarios)

    
    def _bulk_input_preprocess(self) -> None:
        input_obs = self.df['is_obs'].to_numpy()

        ipt_observed_ind_start = np.arange(self.T)[input_obs == True][:-1] + 1
        ipt_observed_ind_start = np.insert(ipt_observed_ind_start, 0, 0)
        ipt_observed_ind_end = np.arange(self.T)[input_obs == True] +1

        sig_r = self.influx[input_obs].std(ddof=1)

        # Bulk case: generate input scenarios based on observed input values

        self.R_prime = np.zeros((self.N, self.T))
        for i in range(sum(input_obs)):
            start_ind = ipt_observed_ind_start[i]
            end_ind = ipt_observed_ind_end[i]

            U = self.influx[start_ind:end_ind].values

            sig_u = self.theta.input_model / (end_ind - start_ind)

            for n in range(self.N):
                self.R_prime[n,start_ind:end_ind] = ss.norm(U, scale=sig_r).rvs()
                if self.R_prime[n,start_ind:end_ind][self.R_prime[n,start_ind:end_ind] >0].size == 0:
                    self.R_prime[n,start_ind:end_ind][self.R_prime[n,start_ind:end_ind] <= 0] = 10**(-8)
                else:
                    self.R_prime[n,start_ind:end_ind][self.R_prime[n,start_ind:end_ind] <= 0] = min(10**(-8), min(self.R_prime[n,start_ind:end_ind][self.R_prime[n,start_ind:end_ind] > 0]))
                U_prime = ss.norm(U[0], sig_u).rvs()
                self.R_prime[n,start_ind:end_ind] = normalize_over_interval(self.R_prime[n,start_ind:end_ind], U_prime)
        
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
        # update input using new sig_r
        if start_ind == 0 and end_ind == 1:
            self._bulk_input_preprocess()

        R_prime = self.R_prime[:,start_ind:end_ind]

        return R_prime   
    
    def observation_model_probability(
        self, yhk: np.ndarray, yk: np.ndarray
    ) -> np.ndarray:
        """get observation probability p(y|y_hat, sig_v)

        Args:
            yhk (np.ndarray): estimated y_hat at time k
            yk (np.ndarray): observation y at time k

        Returns:
            np.ndarray: the likelihood of observation
        """

        theta = self.theta.observation_model
        if len(yhk.shape) == 1:
            return ss.norm(yhk, theta).logpdf(yk)
        else:
            theta /= yhk.shape[1]
            return ss.norm(yhk.mean(axis=1), theta).logpdf(yk)

class ModelInterfaceBulkFineInput(ModelInterface):
    def __init__(self, df: pd.DataFrame, customized_model: Any | None = None, theta_init: dict[str, Any] | None = None, config: dict[str, Any] | None = None, num_input_scenarios: int | None = 10) -> None:
        # initialize the model interface
        super().__init__(df, customized_model, theta_init, config, num_input_scenarios)

        input_obs = self.df['is_obs'].to_numpy()
        interval_ind = np.where(input_obs==True)[0]
        self.obs_interval = interval_ind[-1] - interval_ind[-2]

    def observation_model_probability(
        self, yhk: np.ndarray, yk: np.ndarray
    ) -> np.ndarray:
        """get observation probability p(y|y_hat, sig_v)

        Args:
            yhk (np.ndarray): estimated y_hat at time k
            yk (np.ndarray): observation y at time k

        Returns:
            np.ndarray: the likelihood of observation
        """

        theta = self.theta.observation_model
        if len(yhk.shape) == 1:
            return ss.norm(yhk, theta).logpdf(yk)
        else:
            theta /= self.obs_interval  
            return ss.norm(yhk.mean(axis=1), theta).logpdf(yk)

# %%
class ModelInterfaceDeci(ModelInterface):
    def __init__(self, df: pd.DataFrame, customized_model: Any | None = None, theta_init: dict[str, Any] | None = None, config: dict[str, Any] | None = None, num_input_scenarios: int | None = 10) -> None:
        # initialize the model interface
        super().__init__(df, customized_model, theta_init, config, num_input_scenarios)

        # Decimation case: generate input scenarios based on all input values
    def _deci_input_preprocess(self) -> None:
        not_obs_ipt_ind = np.arange(self.T)[self.df['is_obs'] == False]

        data = self.influx.to_numpy()

        mean = data[~not_obs_ipt_ind].mean()
        std = data[~not_obs_ipt_ind].std(ddof=1)

        # generate random input scenarios for unoberseved input values     
        self.deci_ipt = np.zeros((self.N, self.T))
        self.deci_ipt[:, ~not_obs_ipt_ind] = data[~not_obs_ipt_ind]
        self.deci_ipt[:, not_obs_ipt_ind] = ss.norm(mean, std).rvs((self.N, not_obs_ipt_ind.size))

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
        # update input for new scenarios
        if start_ind == 0 and end_ind == 1:
            self._deci_input_preprocess()
            

        sig_r = self.theta.input_model # ipt obs uncertainty
        R_hat = self.deci_ipt[:,start_ind:end_ind] # actual ipt + guessed ipt
        
        R_prime = np.zeros((self.N, end_ind - start_ind))

        for n in range(self.N):
            R_prime[n,:] = ss.norm(R_hat[n,:], scale=sig_r).rvs(end_ind - start_ind)

        return R_prime

    
# %%
class ModelInterfaceDeciFineInput(ModelInterface):
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

        sig_r = self.theta.input_model # ipt obs uncertainty
        R_hat = self.influx.to_numpy()[start_ind:end_ind] # actual ipt + guessed ipt
        
        R_prime = np.zeros((self.N, end_ind - start_ind))

        for n in range(self.N):
            R_prime[n,:] = ss.norm(R_hat, scale=sig_r).rvs(end_ind - start_ind)

        return R_prime
# %%
