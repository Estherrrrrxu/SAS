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
class ModelInterface(ModelInterface):
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


# %%
class ModelInterfaceWN(ModelInterface):
    def update_model(
            self,
            theta_new: Optional[List[float]] = None
        ) -> None:       
        if theta_new is None:
            theta_new = []
            for key in self._theta_to_estimate:
                theta_new.append(self.dist_model[key].rvs())

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
    
    def input_model(
            self
        ) -> None:
        """Input model for linear reservoir

        Rt' ~ Exp(Ut)
        multiplier * Rk' = Uk + N(0, theta_r)

        Args:
            Ut (float): forcing at time t

        Returns:
            np.ndarray: Rt
        """
        for n in range(self.N):
            self.R[n,] = ss.expon(scale=self.influx).rvs()
            normalized = normalize_over_interval(self.R[n,:], self.observed_ind, self.influx)
            self.R[n,:] = normalized - ss.norm(0,self.theta.input_model).rvs(self.T)
            self.R[n,:][self.R[n,:] < 0] = 0

        return 


# %%

