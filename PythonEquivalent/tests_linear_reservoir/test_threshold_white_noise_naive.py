# %%
import numpy as np
from dataclasses import dataclass
import scipy.stats as ss
import pandas as pd
from typing import Optional, Any, List
from functions.utils import normalize_over_interval
# %%
# a class to store parameters
@dataclass
class Parameter:
    input_model: np.ndarray
    transition_model: np.ndarray
    observation_model: np.ndarray
    initial_state: np.ndarray
# %%
class ModelInterface:
    """Customize necessary model functionalities here

    Methods:
        _parse_config: parse configurations
        _set_observed_made_each_step: set observation interval
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
        """Initialize the model interface, and parse any parameters

        Args:
            df (pd.DataFrame): dataframe of input data
            customized_model (Optional, Any): any model structure that is necessary to import info. Refer to Linear_reservoir for example.
            theta_init (Optional, dict): initial values of parameters
            config (Optional, dict): configurations of the model
            num_input_scenarios (Optional, int): number of input scenarios
        
        Set:
            df (pd.DataFrame): dataframe of input data
            model (Any): any model structure that pass through customized_model. Refer to Linear_reservoir for example. 
            N (int): number of input scenarios
            T (int): number of timesteps
            config (dict): configurations of the model
            R (np.ndarray): input uncertainty storage matrix

        
        """
        self.df = df
        self.model = customized_model # pass your own model here
        self.N = num_input_scenarios
        self.T = len(self.df) # set T to be the length of the dataframe

        # Set configurations according your own need here
        self.config = config
        self._parse_config()

        # Set parameters to be estimated here
        self._parse_theta_init(theta_init=theta_init)

        # initialize input uncertainties
        self.R = np.zeros((self.N, self.T))

        # initialize theta
        self.update_model()

    def _parse_config(self) -> None:
        """Parse config and set default values

        Set:
            dt (float): time step
            config (dict): configurations of the model

            observed_made_each_step (bool or int or List[bool]): whether the input is observed or not. If True, all are observed. If False, ask to pass interval steps as int or observation as a list of bool. If a list of bool, then each entry corresponds to each timestep.   

        """
      
        _default_config = {
            'dt': 1./24/60*15,
            'influx': 'J_obs',
            'outflux': 'Q_obs',
            'observed_made_each_step': True
        }   

        if self.config is None:
            self.config = _default_config

        elif not isinstance(self.config, dict):
            raise ValueError("Error: Please check the input format!")
        else:
            # make sure all keys are valid
            for key in self.config.keys():
                if key not in _default_config.keys():
                    raise ValueError(f'Invalid config key: {key}')
            
            # replace default config with input configs
            for key in _default_config.keys():
                if key not in self.config:
                    self.config[key] = _default_config[key]

        # set delta_t------------------------
        self.dt = self.config['dt']

        # get observation interval set
        self._set_observed_made_each_step()
        # get fluxes set
        self._set_fluxes()

        return
    
    def _set_observed_made_each_step(self) -> None:
        """Set observation interval

        Set:
            K (int): number of observed timesteps
            observed_ind (np.ndarray): indices of observed timesteps
        """
        obs_made = self.config['observed_made_each_step']
        
        # if giving a single bool val
        if isinstance(obs_made, bool):
            # if bool val is True
            if obs_made:
                self.K = self.T
                self.observed_ind = np.arange(self.T)
            # if bool val is False, request to give a int val to give observation interval
            else:
                raise ValueError("Error: Please specify the observation interval!")
            
        # give observation interval as a list
        elif isinstance(obs_made, np.ndarray) or isinstance(obs_made, list):
            # convert np.ndarray to list
            if isinstance(obs_made, np.ndarray):
                obs_made = obs_made.tolist()
            
            # check if the length of the list is the same as the length of the time series
            if len(obs_made) == self.T:
                # if is all bool and all True
                if all(isinstance(entry, bool) and entry for entry in obs_made):
                    self.K = self.T
                    self.observed_ind = np.arange(self.T)

                # if is all bool and some are not True
                elif all(isinstance(entry, bool) for entry in obs_made):
                    self.K = sum(obs_made)
                    self.observed_ind = np.arange(self.T)[obs_made]

            else:
                raise ValueError("Error: Check the format of your observation indicator!")
            
        # give observation interval as a int - how many timesteps
        elif isinstance(obs_made, int):
            # in this case K is still equal to T
            self.K = self.T
            self.config['dt'] *= obs_made # update dt to account for the interval
            self.observed_ind = np.arange(self.K)
        else:
            raise ValueError("Error: Input format not supported!")
    
    def _set_fluxes(self) -> None:
        """Set influx and outflux

        Set:
            influx (np.ndarray): influx
            outflux (np.ndarray): outflux
        """
        # TODO: flexible way to insert multiple influx and outflux
        if self.config['influx'] is not None:
            self.influx = self.df[self.config['influx']]
        else:
            print("Warning: No influx is given!")

        if self.config['outflux'] is not None:
            self.outflux = self.df[self.config['outflux']]
        else:
            print("Warning: No outflux is given!")

        return
    
    def _parse_theta_init(
            self,
            theta_init: Optional[dict[str, Any]] = None            
    ) -> None:
        """Parse theta from theta_init

        Args:
            theta_init (dict): initial values of theta
        """   
        _default_theta_init = {
                'to_estimate': {'k':{"prior_dis": "normal", 
                                     "prior_params":[1.2, 0.3], 
                                     "is_nonnegative": True
                                    },
                                'initial_state':{"prior_dis": "normal", 
                                                 "prior_params":[self.df[self.config['outflux']][0], 0.00005],
                                                 "is_nonnegative": True
                                    },
                                'input_uncertainty':{"prior_dis": "normal", 
                                                     "prior_params":[0.0,0.005],
                                                     "is_nonnegative": False
                                    },
                                },
                'not_to_estimate': {'obs_uncertainty':0.00005}
            }
        self._theta_init = theta_init
        # set default theta
        if theta_init == None:
            self._theta_init = _default_theta_init

        elif not isinstance(theta_init, dict):
            raise ValueError("Error: Please check the input format!")
        else:
            # make sure all keys are valid
            for key in self._theta_init['to_estimate'].keys():
                if key not in _default_theta_init['to_estimate'].keys():
                    print(f'key: {key} not in default config!')
                    # raise ValueError(f'Invalid config key: {key}')
            
            # replace default config with input configs
            for key in _default_theta_init['to_estimate'].keys():
                if key not in self._theta_init['to_estimate']:
                    self._theta_init['to_estimate'][key] = _default_theta_init['to_estimate'][key]
            
        # find params to update
        self._theta_to_estimate = list(self._theta_init['to_estimate'].keys())
        self._num_theta_to_estimate = len(self._theta_to_estimate)

        # Set parameter constraints and distributions
        self._set_parameter_constraints()
        self._set_parameter_distribution()  
        return
    
    def _set_parameter_constraints(self) -> None:
        """Set parameter constraints

        Set:
            param_constraints (dict): whether this parameter is nonnegative
        """
        # save models
        self.param_constraints = {}

        for key in self._theta_to_estimate:
            current_theta = self._theta_init['to_estimate'][key]
            if 'is_nonnegative' in current_theta:
                self.param_constraints[key] = current_theta['is_nonnegative']
            else:
                self.param_constraints[key] = False
        return

    def _set_parameter_distribution(self, update: Optional[bool]=False) -> None:
        """Set prior and update distributions for parameters

        Set:
            dist_model (dict): parameter distribution model
        """
        if not update:
            # save models
            self.dist_model = {}

        for key in self._theta_to_estimate:
            current_theta = self._theta_init['to_estimate'][key]
            is_nonnegative = self.param_constraints[key]
            # set parameters for prior distribution: [first param: mean, second param: std]
            if current_theta['prior_dis'] == 'normal':

                # update or not
                if not update:
                    mean = current_theta['prior_params'][0]
                    
                else:
                    mean = current_theta['current_value']
                std = current_theta['prior_params'][1]
                # truncate or not
                if is_nonnegative:
                    a = (0 - mean) / std
                    self.dist_model[key] = ss.truncnorm(a=a, b=np.inf, loc=mean, scale=std)      
                else:
                    
                    self.dist_model[key] = ss.norm(loc=mean, scale=std)

            elif current_theta['prior_dis'] == 'uniform': # first param: lower bound, second param: upper bound
                # Note: uniform distribution is non-informative prior, so we don't update it
                lower_bound = current_theta['prior_params'][0]
                interval_length = current_theta['prior_params'][1] - current_theta['prior_params'][0]
                self.dist_model[key] = ss.uniform(loc=lower_bound, scale=interval_length)

            else:
                raise ValueError("This prior distribution is not implemented yet")
            
        return
    

    def update_model(
            self,
            theta_new: Optional[List[float]] = None
        ) -> None:       
        """Update the model object with a new parameter set
        
        Set/reset theta to the model
        
        Args:
            theta_new (Optional, Theta): new theta to initialize/update the model
        Set:
            Parameter: update parameter object for later
        """
        if theta_new is None:
            theta_new = []
            for key in self._theta_to_estimate:
                val = self.dist_model[key].rvs()
                theta_new.append(val)
                
        for i, key in enumerate(self._theta_to_estimate):
            self._theta_init['to_estimate'][key]['current_value'] = theta_new[i]
        # transition model param [0] is k to estimate, and [1] is fixed dt
        transition_param = [self._theta_init['to_estimate']['k']['current_value'], self.config['dt']]

        # observation uncertainty param is to estimate
        obs_param = self._theta_init['to_estimate']['obs_uncertainty']

        # input uncertainty param is to estimate
        input_param = self._theta_init['to_estimate']['input_uncertainty']['current_value']

        # initial state param is to estimate
        init_state = self._theta_init['to_estimate']['initial_state']['current_value']

        self.theta = Parameter(
                            input_model=input_param,
                            transition_model=transition_param, 
                            observation_model=obs_param,
                            initial_state=init_state
                            )
        return 
    
    def update_parameter_distribution(
            self
        ) -> dict[str, Any]:
        """Update parameter distribution for SAEM
        """
        self._set_parameter_distribution(update=True)
        return self.dist_model

    
    
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
            Xt[:,i] = (1 -  theta_k * theta_dt) * Xt[:,i-1] + theta_dt * Rt[:,i-1]
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