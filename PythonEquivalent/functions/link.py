# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Any
import scipy.stats as ss

# %%
class ModelLink:
    def __init__(
            self, 
            df: pd.DataFrame, 
            config: Optional[dict[str, Any]] = None,
            theta_init: Optional[dict[str, Any]] = None,
            num_input_scenarios: Optional[int] = 10,
            ) -> None:
        self.df = df
        self.N = num_input_scenarios
        self.config = config

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
        else:
            self.config = self._default_config
            
        # set delta_t------------------------
        self.dt = self.config['dt']
     
        # TODO: flexible way to insert multiple influx and outflux
        self.influx = self.df[self.config['influx']]
        self.outflux = self.df[self.config['outflux']]
        self.T = len(self.influx)
        # set observation interval-----------
        if self.config['observed_at_each_time_step'] == True:
            self.K = self.T
        elif self.config['observed_interval'] is not None:
            self.K = int(self.T/self.config['observed_interval'])
        else:
            self.K = sum(self.config['observed_series']) # total num observation is the number 
            self._is_observed = self.config['observed_series'] # set another variable to indicate observation


        if theta_init == None:
            # set default theta
            theta_init = {
                'to_estimate': {'k':{"prior_dis": "normal", "prior_params":[1.2,0.3], 
                                        "update_dis": "normal", "update_params":[0.05]
                                    },
                                'output_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.0005], 
                                        "update_dis": "normal", "update_params":[0.00001],
                                    }
                                },
                'not_to_estimate': {'input_uncertainty': 0.254*1./24/60*15}
            }
        self._theta_init = theta_init
        # find params to update
        self._theta_to_estimate = list(self._theta_init['to_estimate'].keys())
        self._num_theta_to_estimate = len(self._theta_to_estimate )
        
        # save models
        self.prior_model = {}
        self.update_model = {}

        for key in self._theta_to_estimate:
            current_theta = self._theta_init['to_estimate'][key]
            # for prior distribution
            if current_theta['prior_dis'] == 'normal':
                self.prior_model[key] = ss.norm(loc = current_theta['prior_params'][0], 
                                                scale = current_theta['prior_params'][1])
            elif current_theta['prior_dis'] == 'uniform':
                self.prior_model[key] = ss.uniform(loc = current_theta['prior_params'][0],
                                                    scale = (current_theta['prior_params'][1] - current_theta['prior_params'][0]))
            else:
                raise ValueError("This prior distribution is not implemented yet")
            
            # for update distributions
            if current_theta['update_dis'] == 'normal':
                self.update_model[key] = ss.norm(loc = 0, scale = current_theta['update_params'][0])
            else:
                raise ValueError("This search distribution is not implemented yet")


    def input_model(self, Ut:float) -> np.ndarray:
        """Input model for linear reservoir

        Rt = Ut - U(0, U_upper)

        Args:
            Ut (float): forcing at time t

        Returns:
            np.ndarray: Rt
        """
        input_theta = self._theta_init['not_to_estimate']['input_uncertainty']
        return ss.uniform(Ut-input_theta,input_theta).rvs(self.N)

    def transition_model(self, xtm1: np.ndarray, rt: float, theta_k) -> np.ndarray:
        """Transition model for linear reservoir

        xt = (1 - k * delta_t) * x_{t-1} + k * delta_t * rt
        
        Args:
            xtm1 (np.ndarray): state at t-1
            rt (float): uncertainty r at t

        Returns:
            np.ndarray: xt
        """
        xt = (1 - theta_k * self.dt) * xtm1 + theta_k * self.dt * rt
        return xt
    
    def observation_model(self, xk: np.ndarray) -> np.ndarray:
        """Observation model for linear reservoir

        y_hat(k) = x(k)

        Args:
            xk (np.ndarray): observed y at time k

        Returns:
            np.ndarray: y_hat
        """
        return xk
    
    def f_theta(self, xtm1: np.ndarray, ut: np.ndarray, theta_k) -> np.ndarray:
        """Call transition_model directly

        Args:
            xtm1 (np.ndarray): state at t-1
            rt (float): uncertainty r at t

        Returns:
            np.ndarray: xt
        """
        return self.transition_model(xtm1, ut, theta_k)

    def g_theta(self, yht: np.ndarray, yt: np.ndarray, theta_obs) -> np.ndarray:
        """Observation model for linear reservoir

        y(t) = y_hat(t) + N(0, theta_v)

        Args:
            yht (np.ndarray): estimated y_hat at time t
            yt (np.ndarray): observed y at time t
            theta (float): parameter of the model

        Returns:
            np.ndarray: p(y|y_hat, sig_v)
        """
        return ss.norm(yht, theta_obs).pdf(yt)