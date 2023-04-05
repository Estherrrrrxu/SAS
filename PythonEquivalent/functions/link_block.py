# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import scipy.stats as ss
from estimator_block import *

# %%
class LinearReservoirInputModel(InputModel):
    """
    Args:
        theta (np.ndarray): [U_upper, N]
    """
    def input(self, Ut:float) -> np.ndarray:
        """Input model for linear reservoir

        Rt = Ut - U(0, U_upper)

        Args:
            Ut (float): forcing at time t

        Returns:
            np.ndarray: Rt
        """
        return ss.uniform(Ut-self.theta[0],self.theta[0]).rvs(self.theta[1])

class LinearReservoirTranModel(TransitionModel):
    """
    Args:
        theta (np.ndarray): [k, delta_t]
    """
    def transition(self, xtm1: np.ndarray, rt: float) -> np.ndarray:
        """Transition model for linear reservoir

        xt = (1 - k * delta_t) * x_{t-1} + k * delta_t * rt
        
        Args:
            xtm1 (np.ndarray): state at t-1
            rt (float): uncertainty r at t

        Returns:
            np.ndarray: xt
        """
        xt = (1 - self.theta[0] * self.theta[1]) * xtm1 + self.theta[0] * self.theta[1] * rt
        return xt
    
class LinearReservoirObsModel(ObservationModel):
    def observation(self, xk: np.ndarray) -> np.ndarray:
        """Observation model for linear reservoir

        y_hat(k) = x(k)

        Args:
            xk (np.ndarray): observed y at time k

        Returns:
            np.ndarray: y_hat
        """
        return xk
    
class LinearReservoirProposalModel(ProposalModel):
    """_summary_

    Args:
        theta (np.array): sig_v
    """
    def f_theta(self, xtm1: np.ndarray, ut: np.ndarray) -> np.ndarray:
        """Call transition_model directly

        Args:
            xtm1 (np.ndarray): state at t-1
            rt (float): uncertainty r at t

        Returns:
            np.ndarray: xt
        """
        return self.transition_model.transition(xtm1, ut)

    def g_theta(self, yht: np.ndarray, yt: np.ndarray) -> np.ndarray:
        """Observation model for linear reservoir

        y(t) = y_hat(t) + N(0, theta_v)

        Args:
            yht (np.ndarray): estimated y_hat at time t
            yt (np.ndarray): observed y at time t
            theta (float): parameter of the model

        Returns:
            np.ndarray: p(y|y_hat, sig_v)
        """
        return ss.norm(yht, self.theta).pdf(yt)
    
class LinearReservoirInputProcess(InputProcess):
    def _process_config(self) -> None:
        # process configurations------------
        self._default_config = {
            'dt': 1./24/60*15,
            'influx': 'J_obs',
            'outflux': 'Q_obs',
            'observed_at_each_time_step': True,
            'observed_interval': None,
            'observed_series': None # give a boolean list of observations been made 
        }   

        # replace default config with input configs
        if self.config is not None:
            for key in self._default_config:
                if key not in self.config:
                    self.config[key] = self._default_config[key]
            else:
                raise ValueError('Invalid config key: {}'.format(key))
        
        # set observation interval-----------
        if self.config['observed_at_each_time_step'] == True:
            self.K = self.T
        elif self.config['observed_interval'] is not None:
            self.K = int(self.T/self.config['observed_interval'])
        else:
            self.K = sum(self.config['observed_series']) # total num observation is the number 
            self._is_observed = self.config['observed_series'] # set another variable to indicate observation

        # set delta_t------------------------
        self.dt = self.config['dt']

    def _process_data(self) -> None:       
        # TODO: flexible way to insert multiple influx and outflux
        self.influx = self.df[self.config['influx']]
        self.outflux = self.df[self.config['outflux']]



 # ---------------pGS_SAEM algo----------------
    def _process_theta(self,theta_init=None):
        """
            For theta models:
            1. Prior model: 
                normal:     params [mean, std]
                uniform:    params [lower bound, upper bound]
            2. Update model:
                normal:     params [std]
        """
        if theta_init == None:
            # set default theta
            theta_init = {
                'to_estimate': {'k':{"prior_dis": "normal", "prior_params":[1.2,0.3], 
                                        "update_dis": "normal", "update_params":[0.05],
                                        'log':False},
                                'output_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.0005], 
                                        "update_dis": "normal", "update_params":[0.00005],
                                        'log': True}
                                },
                'not_to_estimate': {'input_uncertainty': 0.254*1./24/60*15}
            }

        # find params to update
        self._theta_to_estimate = list(self._theta_init['to_estimate'].keys())
        self._num_theta_to_estimate = len(self._theta_to_estimate )
        
        # save models
        self.prior_model = {}
        self.update_model = {}

        for key in self._theta_to_estimate:
            current_theta = self._theta_init['to_estimate'][key]
            if current_theta['log'] == True:
                # TODO: need to think about how to do log part
                # for prior distribution
                if current_theta['prior_dis'] == 'normal':
                    self.prior_model[key] = ss.norm(loc = np.log(current_theta['prior_params'][0]), scale = np.log(current_theta['prior_params'][1]))
                elif current_theta['prior_dis'] == 'uniform':
                    self.prior_model[key] = ss.uniform(loc = np.log(current_theta['prior_params'][0]),scale = np.log(current_theta['prior_params'][1] - current_theta['prior_params'][0]))
                else:
                    raise ValueError("This prior distribution is not implemented yet")
                
                # for update distributions
                if current_theta['update_dis'] == 'normal':
                    self.update_model[key] = ss.norm(loc = 0, scale = current_theta['update_params'][0])
                else:
                    raise ValueError("This search distribution is not implemented yet")
            else:
                # for prior distribution
                if current_theta['prior_dis'] == 'normal':
                    self.prior_model[key] = ss.norm(loc = current_theta['prior_params'][0], scale = current_theta['prior_params'][1])
                elif current_theta['prior_dis'] == 'uniform':
                    self.prior_model[key] = ss.uniform(loc = current_theta['prior_params'][0],scale = (current_theta['prior_params'][1] - current_theta['prior_params'][0]))
                else:
                    raise ValueError("This prior distribution is not implemented yet")
                
                # for update distributions
                if current_theta['update_dis'] == 'normal':
                    self.update_model[key] = ss.norm(loc = 0, scale = current_theta['update_params'][0])
                else:
                    raise ValueError("This search distribution is not implemented yet")
        



proposal_model = LinearReservoirProposalModel(
    transition_model = LinearReservoirTranModel(),
    observation_model = LinearReservoirObsModel()
)

input_model = LinearReservoirInputModel(theta = )
num_input_scenarios = 20

LinearReservoirModel = SSModel(
        num_input_scenarios,
        proposal_model,
        input_model
    )






SSM = SSModel(
    input_model,
    transition_model,
    observation_model,
    f_theta = f_theta_for_MESAS,
    g_theta = g_theta_for_MESAS
    )
SSM.run_particle_MCMC(f_theta = f_theta_after_MESAS, g_theta = g_theta_after_MESAS)

