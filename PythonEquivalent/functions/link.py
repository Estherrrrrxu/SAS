# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Any
import scipy.stats as ss
from functions.model_structure import SpecifyModel, LinearReservior, Parameter

# %%
class ModelLink:
    def __init__(
            self, 
            df: pd.DataFrame,
            customized_model: SpecifyModel,
            theta_init: Optional[dict[str, Any]] = None,
            config: Optional[dict[str, Any]] = None,
            num_input_scenarios: Optional[int] = 10,      
            ) -> None:
        """Link model structure and data

        Args:
            df (pd.DataFrame): dataframe of input data
            customized_model (SpecifyModel): model structure
            config (dict): configurations of the model
            num_input_scenarios (int): number of input scenarios
            
        """
        self.df = df
        self.N = num_input_scenarios

        self.config = config
        # customize the config according to model structure here
        self._parse_config()
        # parse theta - add customization here
        self._parse_theta_init(theta_init)

        # initialize model
        self.model = customized_model

    def _parse_config(
            self,
        ) -> None:
        """Parse config and set default values

        """

        # process configurations------------
        _default_config = {
            'dt': 1./24/60*15,
            'influx': 'J_obs',
            'outflux': 'Q_obs',
            'observed_at_each_time_step': True,
            'observed_interval': None,
            'observed_series': None # give a boolean list of observations been made 
        }   
        # replace default config with input configs
        

        if self.config is not None:
            for key in _default_config.keys():
                if key not in self.config:
                    self.config[key] = _default_config[key]
            # else:
            #     raise ValueError(f'Invalid config key: {key}')
        else:
            self.config = _default_config
            
        # set delta_t------------------------
        self.dt = self.config['dt']
     
        # TODO: flexible way to insert multiple influx and outflux
        if self.config['influx'] is not None:
            self.influx = self.df[self.config['influx']]
            self.T = len(self.influx)
        
        self.outflux = self.df[self.config['outflux']]
        
        # set observation interval-----------
        if self.config['observed_at_each_time_step'] == True:
            self.K = self.T
        elif self.config['observed_interval'] is not None:
            self.K = int(self.T/self.config['observed_interval'])
        else:
            self.K = sum(self.config['observed_series']) # total num observation is the number 
            self._is_observed = self.config['observed_series'] # set another variable to indicate observation
        return

    def _parse_theta_init(
            self,
            theta_init: Optional[dict[str, Any]] = None            
    ) -> None:
        """Parse theta and set default values

        Args:
            theta_init (dict): initial values of theta
        """
        if theta_init == None:
            # set default theta
            theta_init = {
                'to_estimate': {'k':{"prior_dis": "normal", "prior_params":[1.2,0.3], 
                                        "update_dis": "normal", "update_params":[0.05]
                                    },
                                'obs_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.0005], 
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
                # need theta be decoded in this way
        return


    def sample_theta_from_prior(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        theta_new = np.zeros(self._num_theta_to_estimate)
        for i, key in enumerate(self._theta_to_estimate):
             theta_new[i] = self.prior_model[key].rvs()
        # cannot update model here, because it need to update iteratively
        self._update_model(theta_new)
        return theta_new

    def _update_model(
            self,
            theta_new: np.ndarray
        ) -> None:
        # create function that updates the model object with a new parameter set
        # and calls whatever initialization routines are necessary        
        """Set theta to the model, customizable
        
        Args:
            theta_new (np.ndarray): new theta to update the model

        Set:
            Parameter: update parameter object for later
        """
        # input model param is fixed
        input_param = self._theta_init['not_to_estimate']['input_uncertainty']
        # transition model param [0] is theta to estimate, and [1] is fixed dt
        transition_param = [theta_new[0], self.config['dt']]
        # observation model param is to estimate
        obs_param = theta_new[1]

        self.theta = Parameter(
                            input_model=input_param,
                            transition_model=transition_param, 
                            observation_model=obs_param
                            )
        self.model = self.model(self.theta, self.N)
        return 
    
    def f_theta(self, 
                xtm1: np.ndarray, 
                ut: np.ndarray
        ) -> np.ndarray:
        """Call transition_model directly

        Args:
            xtm1 (np.ndarray): state at t-1
            rt (float): uncertainty r at t

        Returns:
            np.ndarray: state x at t
        """
        xt = self.model.transition_model(xtm1 = xtm1, ut = ut)
        return xt
    def g_theta(self, 
                xk: np.ndarray,
                yt: np.ndarray
        ) -> np.ndarray:
        """Observation model for linear reservoir

        y(t) = y_hat(t) + N(0, theta_v),
                        where y_hat(t) = x(t) 

        Args:
            xt (np.ndarray): state x at time t
            yt (np.ndarray): observed y at time t

        Returns:
            np.ndarray: p(y|y_hat, sig_v)
        """
        yht = self.model.observation_model(xk = xk)
        return ss.norm(yht).pdf(yt)
    
    def input_generation(self,
        ) -> np.ndarray:
        """Generate input uncertainty for the model
        
        Returns:
            np.ndarray: input for the model"""

        self.R = np.zeros((self.T, self.N))
        for k in range(self.T):
            self.R[k,:] = self.model.input_model(self.influx[k])
        return

