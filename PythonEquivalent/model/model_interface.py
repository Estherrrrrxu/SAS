# %%
import numpy as np
from dataclasses import dataclass
import scipy.stats as ss
import pandas as pd
from typing import Optional, Any, List

# %%
# a class to store parameters
@dataclass
class Parameter:
    input_model: List[float]
    transition_model: List[float]
    observation_model: List[float]
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
        observation_model_likelihood: observation model likelihood for/to link user-defined model
        input_model: input model for/to link user-defined model
        state_as_model: state model for/to link user-defined model
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
            config (dict): configurations of the model
            N (int): number of input scenarios
            T (int): length of the dataframe

        """
        self.df = df
        self.model = customized_model  # pass your own model here
        self.N = num_input_scenarios
        self.T = len(self.df)  # set T to be the length of the dataframe

        # Set configurations according your own need here
        self.config = config
        self._parse_config()

        # Set parameters to be estimated here
        self._parse_theta_init(theta_init=theta_init)

        # initialize theta
        self.update_theta()

        # initialize model
        self.num_states, self.num_obs = None, None

    def _parse_config(self) -> None:
        """Parse config and set default values

        Set:
            dt (float): time step
            config (dict): configurations of the model
        """

        _default_config = {
            "dt": 1.0,
            "influx": "J_obs",
            "outflux": "Q_obs",
            "observed_made_each_step": True,
            "use_MAP_ref_traj": False,
            "use_MAP_AS_weight": False,
            "use_MAP_MCMC": False,
            "update_theta_dist": False
        }

        if self.config is None:
            self.config = _default_config

        elif not isinstance(self.config, dict):
            raise ValueError("Error: Please check the input format!")
        else:
            # make sure all keys are valid
            for key in self.config.keys():
                if key not in _default_config.keys():
                    raise ValueError(f"Invalid config key: {key}")

            # replace default config with input configs
            for key in _default_config.keys():
                if key not in self.config:
                    self.config[key] = _default_config[key]

        # set delta_t------------------------
        self.dt = self.config["dt"]

        # get observation interval set
        self._set_observed_made_each_step()
        # get fluxes set
        self._set_fluxes()

        return

    def _set_observed_made_each_step(self) -> None:
        """Set observation interval based on config

        Set:
            K (int): number of observed timesteps
            observed_ind (np.ndarray): indices of observed timesteps
        """
        obs_made = self.config["observed_made_each_step"]

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
                raise ValueError(
                    "Error: Check the format of your observation indicator!"
                )

        # give observation interval as a int - how many timesteps
        elif isinstance(obs_made, int):
            # in this case K is still equal to T
            
            if self.T % obs_made == 0:
                self.K = int(self.T//obs_made)
            else:
                self.K = int(self.T//obs_made)+1
            self.observed_ind = np.arange(self.T, step=obs_made)

        else:
            raise ValueError("Error: Input format not supported!")

    def _set_fluxes(self) -> None:
        """Set influx and outflux

        Set:
            influx (np.ndarray): influx
            outflux (np.ndarray): outflux
        """
        # TODO: flexible way to insert multiple influx and outflux
        if self.config["influx"] is not None:
            self.influx = self.df[self.config["influx"]]
        else:
            print("Warning: No influx is given!")

        if self.config["outflux"] is not None:
            self.outflux = self.df[self.config["outflux"]]
        else:
            print("Warning: No outflux is given!")

        return

    def _parse_theta_init(self, theta_init: Optional[dict[str, Any]] = None) -> None:
        """Parse theta from theta_init
        new theta_init can overwrite the default theta_init
        default theta_init can fill in the missing keys in new theta_init

        Args:
            theta_init (Optional, dict): initial values of parameters

        Set:
            theta_init (dict): initial values of parameters
        """
        _default_theta_init = {
            "to_estimate": {
                "k": {
                    "prior_dis": "normal",
                    "prior_params": [1.0, 0.0001],
                    "is_nonnegative": True,
                },
                "initial_state": {
                    "prior_dis": "normal",
                    "prior_params": [self.df[self.config["outflux"]][0], 0.0001],
                    "is_nonnegative": True,
                },
                "input_uncertainty": {
                    "prior_dis": "normal",
                    "prior_params": [
                        self.df[self.config["influx"]].std(ddof=1),
                        0.0005,
                    ],
                    "is_nonnegative": True,
                },
                "obs_uncertainty": {
                    "prior_dis": "normal",
                    "prior_params": [
                        self.df[self.config["outflux"]].std(ddof=1),
                        0.000005,
                    ],
                    "is_nonnegative": True,
                },
            },
            "not_to_estimate": {},
        }

        self._theta_init = theta_init
        # set default theta
        if theta_init == None:
            self._theta_init = _default_theta_init

        elif not isinstance(theta_init, dict):
            raise ValueError("Error: Please check the input format!")
        else:
            # make sure all keys are valid
            for key in self._theta_init["to_estimate"].keys():
                if key not in _default_theta_init["to_estimate"].keys():
                    print(f"key: {key} not in default config!")
                    # raise ValueError(f'Invalid config key: {key}')

            # replace default config with input configs
            for key in _default_theta_init["to_estimate"].keys():
                if key not in self._theta_init["to_estimate"]:
                    self._theta_init["to_estimate"][key] = _default_theta_init[
                        "to_estimate"
                    ][key]

        # find params to update
        self._theta_to_estimate = list(self._theta_init["to_estimate"].keys())
        self._num_theta_to_estimate = len(self._theta_to_estimate)

        # Set parameter constraints and distributions
        self._set_parameter_constraints()
        self._set_parameter_distribution()
        return

    def _set_parameter_constraints(self) -> None:
        """Get parameter constraints: nonnegative or not

        Set:
            param_constraints (dict): whether this parameter is nonnegative
        """
        # save models
        self.param_constraints = {}

        for key in self._theta_to_estimate:
            current_theta = self._theta_init["to_estimate"][key]
            if "is_nonnegative" in current_theta:
                self.param_constraints[key] = current_theta["is_nonnegative"]
            else:
                self.param_constraints[key] = False
        return

    def _set_parameter_distribution(
        self, update: Optional[bool] = False, theta_new: Optional[dict] = None
    ) -> None:
        """Set parameter distribution model and update the model object

        Args:
            update (Optional, bool): whether to update the model object
            theta_new (Optional, float): new theta to initialize/update the model

        Set:
            dist_model (dict): parameter distribution model
        """
        if not update:
            # save models
            self.dist_model = {}

        for key in self._theta_to_estimate:
            # grab theta
            current_theta = self._theta_init["to_estimate"][key]
            is_nonnegative = self.param_constraints[key]

            # set parameters for prior distribution: [first param: mean, second param: std]
            if current_theta["prior_dis"] == "normal":
                # update or not
                if not update:
                    mean, std = current_theta["prior_params"]

                else:
                    mean, std = theta_new[key]
                    if std < mean/10.:
                        std = mean/10.         

                # truncate or not
                if is_nonnegative:
                    a = (0 - mean) / std
                    self.dist_model[key] = ss.truncnorm(
                        a=a, b=np.inf, loc=mean, scale=std
                    )
                else:
                    self.dist_model[key] = ss.norm(loc=mean, scale=std)

            elif current_theta["prior_dis"] == "uniform":
                # first param: lower bound, second param: upper bound
                # Note: uniform distribution is non-informative prior, so we don't update it
                lower_bound = current_theta["prior_params"][0]
                interval_length = (
                    current_theta["prior_params"][1] - current_theta["prior_params"][0]
                )
                self.dist_model[key] = ss.uniform(
                    loc=lower_bound, scale=interval_length
                )

            else:
                raise ValueError("This prior distribution is not implemented yet")

        return

    def update_theta(self, theta_new: Optional[List[float]] = None) -> None:
        """Set/update the model object with a new parameter set

        Args:
            theta_new (Optional, Theta): new theta to initialize/update the model

        Set:
            Parameter: update parameter object for later
        """
        # initialize theta to the order of self._theta_to_estimate
        if theta_new is None:
            theta_new = np.zeros(self._num_theta_to_estimate)
            for i, key in enumerate(self._theta_to_estimate):
                theta_new[i] = self.dist_model[key].rvs()

        # if a set of theta is given, update the current theta to the new theta
        for i, key in enumerate(self._theta_to_estimate):
            self._theta_init["to_estimate"][key]["current_value"] = theta_new[i]

        # update parameters to be passed into the model

        # transition model
        transition_param = [
            self._theta_init["to_estimate"]["k"][
                "current_value"
            ],  # param [0] is k to estimate
            self.config["dt"],  # param [1] is delta_t (fixed)
        ]

        # observation model
        obs_param = self._theta_init["to_estimate"]["obs_uncertainty"]["current_value"]

        # input uncertainty param is to estimate
        input_param = self._theta_init["to_estimate"]["input_uncertainty"][
            "current_value"
        ]

        # initial state param is to estimate
        init_state = self._theta_init["to_estimate"]["initial_state"]["current_value"]

        # update model object theta
        self.theta = Parameter(
            input_model=input_param,
            transition_model=transition_param,
            observation_model=obs_param,
            initial_state=init_state,
        )
        return

    def input_model(self, start_ind: int, end_ind: int) -> None:
        """Input model for linear reservoir


        Ut: influx at time t
        Rt = N(Ut, theta_r)

        Args:
            start_ind (int): start index of the input time series
            end_ind (int): end index of the input time series

        Returns:
            np.ndarray: Rt
        """

        sig_r = self.theta.input_model
        R = np.zeros((self.N, end_ind - start_ind))
        U = self.influx[start_ind:end_ind] 

        for n in range(self.N):
            R[n] = ss.norm(U, scale=sig_r).rvs()

        return R

    def transition_model(
        self, Xtm1: np.ndarray, Rt: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """State estimaton model f_theta(Xt-1, Rt)

        Currently set up for linear reservoirmodel:
            Xt = (1 - k * delta_t) * X_{t-1} + delta_t * Rt,
                where Rt = N(Ut, theta_r) from input model

        Args:
            Xtm1 (np.ndarray): state X at t = k-1
            Rt (np.ndarray, Optional): input signal at t = k-1:k

        Returns:
            np.ndarray: state X at t
        """
        # Get parameters
        theta = self.theta.transition_model
        theta_k = theta[0]
        theta_dt = theta[1]

        # update from last observation
        num_iter = Rt.shape[1]
        Xt = np.ones((self.N, num_iter + 1)) * Xtm1.reshape(-1, 1)

        for i in range(1, num_iter + 1):
            Xt[:, i] = (1 - theta_k * theta_dt) * Xt[:, i - 1] + theta_k * Rt[:, i - 1]

        return Xt[:, 1:]  # return w/out initial state

    def transition_model_probability(self, X_1toT: np.ndarray) -> np.ndarray:
        """State estimaton model f_theta(Xt-1, Rt)

        Currently set up for linear reservoirmodel:
            p(Xt|Xtm1) = (1 - k * delta_t) * Xtm1 + delta_t * Rt,
                where Rt = N(Ut, theta_r) from input model
            p(Xt|Xtm1) = N((1 - k * delta_t) * Xtm1 + delta_t * Ut, delta_t * theta_r)    

        Args:
            X_1toT (np.ndarray): state X at t = 1:T

        Returns:
            np.ndarray: p(X_{1:T}|theta)
        """
        # # Get parameters
        # theta = self.theta.transition_model
        # theta_k = theta[0]
        # theta_dt = theta[1]
        # theta_r = self.theta.input_model

        # # set all params
        # prob = np.ones((self.N, self.T - 1))
        # Ut = self.influx[1:].to_numpy()
        # Xtm1 = X_1toT[:-1]
        # Xt = X_1toT[1:]

        # # calculate prob
        # prob = ss.norm(
        #     (1 - theta_k * theta_dt) * Xtm1 + theta_dt * Ut, theta_r * (theta_dt)
        # ).logpdf(Xt) 

        # return prob.sum()
        return 1.

    def observation_model(self, Xk: np.ndarray) -> np.ndarray:
        """Observation probability g_theta

        Current setup for linear reservoir:
            y_hat(k) = x(k)

        Current general model setting:
            y(t) = y_hat(t) - N(0, theta_v),
                            where y_hat(t) = x(t)

        Args:
            Xk (np.ndarray): state X at time k
            yhk (np.ndarray): estimated y_hat at time k

        Returns:
            np.ndarray: y_hat at time k
        """

        return Xk

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
            return ss.norm(yhk[:, -1], theta).logpdf(yk)
    
    def initial_state_model(self, num: int) -> np.ndarray:
        """Initial state model

        Args:
            num (int): number of initial states to generate

        Returns:
            np.ndarray: initial state
        """

        return self.dist_model["initial_state"].rvs(num).reshape(-1, 1)

    def state_as_probability(self, offset: np.ndarray, std: float):
        """State model probability p(xk'|xk)

        Args:
            offset (np.ndarray): offset of state model (xk - xk')
            sd (float): standard deviation of state model, set to be adaptive for now

        Returns:
            np.ndarray: probability of estimated state around ref trajectory
        """

        return ss.norm(0, std).logpdf(offset)


# %%
