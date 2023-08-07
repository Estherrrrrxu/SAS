import numpy as np
from dataclasses import dataclass
import scipy.stats as ss
import pandas as pd
from typing import Optional, Any, List
from functions.utils import normalize_intervals
# %%
@dataclass
class Parameter:
    input_model: np.ndarray
    transition_model: np.ndarray
    observation_model: np.ndarray
# %%
class ModelInterface:
    """Customize necessary model functionalities here

    Methods:
        _parse_config: parse configurations
        _set_fluxes: set fluxes and observations using info from config
        _parse_theta_init: parse initial values of parameters
        _set_parameter_distribution: set parameter distribution using info from theta_init

        update_model: update model using the new theta values
        transition_model: transition model for/to link user-defined model
        observation_model: observation model for/to link user-defined model


    """
    def __init__(
            self, 
            df: pd.DataFrame,
            customized_model: Optional[Any] = None,
            theta_init: Optional[dict[str, Any]] = None,
            config: Optional[dict[str, Any]] = None,
            num_input_scenarios: Optional[int] = 10,   
        ) -> None:
        """Initialize the model interface

        Args:
            df (pd.DataFrame): dataframe of input data
            customized_model (Any): any model structure that is necessary to import info
            theta_init (dict): initial values of parameters
            config (dict): configurations of the model
            num_input_scenarios (int): number of input scenarios
            observed_made_each_step (bool or int or List[bool]): whether the input is observed or not
            observed_ind (np.ndarray): mapping from T to K
        
        """
        self.df = df
        self.N = num_input_scenarios

        self.config = config
        # customize the config according to model structure here
        self._parse_config()
        self._set_fluxes()
        # parse theta - add customization here
        self._parse_theta_init(theta_init=theta_init)
        self._set_parameter_distribution()
        # initialize model
        self.model = customized_model
        # initialize input uncertainties
        self.R = np.zeros((self.N, self.T))
        self.update_model([1, 0.0005, 0.0005]) # dummy update

        self.initial_state = self.outflux[0]

    def _parse_config(
            self,
        ) -> None:
        """Parse config and set default values

        """

        # process configurations------------
        """
        dt: float
        influx: str
        outflux: str
        observed_made_each_step: bool or int or List[bool]
        """
        _default_config = {
            'dt': 1./24/60*15,
            'influx': 'J_obs',
            'outflux': 'Q_obs',
            'observed_made_each_step': True,
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
        return
    
    def _set_fluxes(self) -> None:
        """Set fluxes
        """
        # TODO: flexible way to insert multiple influx and outflux
        if self.config['influx'] is not None:
            self.influx = self.df[self.config['influx']]
            self.T = len(self.influx)
        
        self.outflux = self.df[self.config['outflux']]

        obs_made = self.config['observed_made_each_step']
        
        # set observation K -----------
        # if giving a single bool val
        if isinstance(obs_made, bool):
            # if bool val is True
            if obs_made:
                self.K = self.T
                self.observed_ind = [True] * self.T
            # if bool val is False, request to give a int val to give observation interval
            else:
                raise ValueError("Error: Please specify the observation interval!")
        # give observation list
        elif isinstance(obs_made, np.ndarray) or isinstance(obs_made, list):
            # convert np.ndarray to list
            if isinstance(obs_made, np.ndarray):
                obs_made = obs_made.tolist()
            # if is all bool and all True
            if all(isinstance(entry, bool) and entry for entry in obs_made):
                self.K = self.T
                self.observed_ind = np.arange(self.T)
            # if is all bool and some are not True
            elif all(isinstance(entry, bool) for entry in obs_made):
                self.K = sum(obs_made)
                self.observed_ind = np.arange(self.T)[obs_made]
            else:
                raise ValueError("Error: Invalid input!")
        # give observation interval as a int - how many timesteps
        elif isinstance(obs_made, int):
            # in this case K still = T
            self.K = self.T
            self.config['dt'] *= obs_made # update dt
            self.observed_ind = np.arange(self.K)
        else:
            raise ValueError("Error: Please check the input format!")
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
                                        "search_dis": "normal", "search_params":[0.05]
                                    },
                                'obs_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.0005], 
                                        "search_dis": "normal", "search_params":[0.00001],
                                    }
                                },
                'not_to_estimate': {'input_uncertainty': 0.254*1./24/60*15}
            }
        self._theta_init = theta_init
        # find params to update
        self._theta_to_estimate = list(self._theta_init['to_estimate'].keys())
        self._num_theta_to_estimate = len(self._theta_to_estimate )
        return
    
    def _set_parameter_distribution(
            self,
        ) -> None:
        """Set prior and update distributions for parameters

        """
        # save models
        self.prior_model = {}
        self.search_model = {}

        for key in self._theta_to_estimate:
            current_theta = self._theta_init['to_estimate'][key]
            # for prior distribution
            if current_theta['prior_dis'] == 'normal':
                self.prior_model[key] = ss.norm(
                    loc = current_theta['prior_params'][0], 
                    scale = current_theta['prior_params'][1]
                                        )
            elif current_theta['prior_dis'] == 'uniform':
                # first param: lower bound, second param: interval length
                self.prior_model[key] = ss.uniform(
                    loc = current_theta['prior_params'][0],
                    scale = (
                    current_theta['prior_params'][1] - current_theta['prior_params'][0]
                            )
                                                )
            else:
                raise ValueError("This prior distribution is not implemented yet")
            
            # for update distributions
            if current_theta['search_dis'] == 'normal':
                self.search_model[key] = ss.norm(loc = 0, scale = current_theta['search_params'][0])
            else:
                raise ValueError("This search distribution is not implemented yet")
                # need theta be decoded in this way

    def update_model(
            self,
            theta_new: np.ndarray
        ) -> None:       
        """Update the model object with a new parameter set
        
        Set/reset theta to the model
        
        Args:
            theta_new (np.ndarray): new theta to update the model

        Set:
            Parameter: update parameter object for later
        """


        # transition model param [0] is theta to estimate, and [1] is fixed dt
        transition_param = [theta_new[0], self.config['dt']]
        # observation model param is to estimate
        obs_param = theta_new[1]

        # input model param is to estimate
        input_param = theta_new[2]

        self.theta = Parameter(
                            input_model=input_param,
                            transition_model=transition_param, 
                            observation_model=obs_param
                            )
        return 
    
    def transition_model(self, 
                        Xtm1: np.ndarray, 
                        Rt: Optional[np.ndarray] = None
        ) -> np.ndarray:
        """State estimaton model f_theta(Xt-1, Rt)

        Currently set up for linear reservoirmodel:
            xt = (1 - k * delta_t) * x_{t-1} + k * delta_t * rt,
                where rt = ut - U(0, theta_r)

        Args:
            Xtm1 (np.ndarray): state X at t = k-1
            Rt (np.ndarray, Optional): uncertainty R at t = k-1:k

        Returns:
            np.ndarray: state X at t
        """
        # Get parameters
        theta = self.theta.transition_model
        theta_k = theta[0]
        theta_dt = theta[1]
        
        # update from last observation
        num_iter = Rt.shape[1]
        Xt = np.ones((self.N, num_iter+1)) * Xtm1.reshape(-1, 1)        
        for i in range(1,num_iter+1):
            Xt[:,i] = (1 -  theta_k * theta_dt) * Xt[:,i-1] + theta_k * theta_dt * Rt[:,i-1]
        return Xt[:,1:]
    

    def observation_model(self, 
                        Xk: np.ndarray
        ) -> np.ndarray:
        """Observation likelihood g_theta

        Current setup for linear reservoir:
            y_hat(k) = x(k)

        Current general model setting:
            y(t) = y_hat(t) + N(0, theta_v),
                            where y_hat(t) = x(t) 

        Args:
            Xk (np.ndarray): state X at time k
            yhk (np.ndarray): estimated y_hat at time k

        Returns:
            np.ndarray: y_hat at time k
        """
        yhk = Xk
        return yhk
    
    def observation_model_likelihood(self,
                                    yhk: np.ndarray,
                                    yk: np.ndarray
        ) -> np.ndarray:
        """get observation likelihood p(y|y_hat, sig_v)

        Args:
            yhk (np.ndarray): observation y_hat at time k
            yk (np.ndarray): observation y at time k

        Returns:
            np.ndarray: the likelihood of observation
        """
        theta = self.theta.observation_model
        return ss.norm(yhk, theta).pdf(yk)

    
    def input_model(
            self
        ) -> None:
        """Input model for linear reservoir

        Rt = Ut - U(0, U_upper)

        Args:
            Ut (float): forcing at time t

        Returns:
            np.ndarray: Rt
        """

        for t in range(self.T):
            # self.R[:,t] = ss.uniform(
            #     self.influx[t] - self.theta.input_model, self.theta.input_model
            #     ).rvs(self.N)
            self.R[:,t] = ss.expon(scale=self.influx[t]).rvs(self.N)
        for n in range(self.N):
            self.R[n,:] = normalize_intervals(self.R[n,:], self.observed_ind)
            self.R[n,:] *= (self.influx - ss.norm(0,self.theta.input_model).rvs(self.T))
            self.R[n,:][self.R[n,:] < 0] = 0

        return 
    
    def state_model(
            self,
            x_prime: float,
            xkp1: np.ndarray,
            sd: float
    ):
        """State model for linear reservoir
        
        Args:
            x_prime (float): state of ref trajectory at time t+1
            xkp1 (np.ndarray): estimated state at time t+1
            sd (float): standard deviation of state model, set to be adaptive for now
            
        Returns:
            np.ndarray: likelihood of estimated state around ref trajectory
        """
        # sd = self.theta.state_model
        return ss.norm(x_prime, sd).pdf(xkp1)

    
