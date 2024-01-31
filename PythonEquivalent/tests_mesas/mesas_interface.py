# %%
import os

current_path = os.getcwd()

if current_path[-11:] != "tests_mesas":
    os.chdir("tests_mesas")
    print("Current working directory changed to 'tests_mesas'.")
import sys

sys.path.append("../")
import numpy as np
from dataclasses import dataclass
import scipy.stats as ss
import pandas as pd
from typing import Optional, Any, List
from mesas.sas.model import Model as SAS_Model
from functions.utils import normalize_over_interval
from copy import deepcopy


# %%
# a class to store parameters
@dataclass
class Parameter:
    input_model: List[float]
    transition_model: List[float]
    observation_model: List[float]
    initial_state: np.ndarray


# %%
class ParamsProcessor:
    def __init__(self, distinct_str):
        self.params = {"to_estimate": {}, "not_to_estimate": {}}
        self.distinct_str = distinct_str
        self.transit_params = {"to_estimate": [], "not_to_estimate": []}
        self.init_state_params = []

    def setup_prior_params(self, mode, flux, sas_name, param_key, sas_func):
        name = self.distinct_str.join([flux, sas_name, param_key])

        # if this parameter is not to be estimated
        if mode == "not_to_estimate":
            self.params["not_to_estimate"][name] = {sas_func["args"][param_key]}
            self.transit_params["not_to_estimate"].append(name)
        elif mode == "to_estimate":
            is_C_old = param_key == "C_old"
            is_prior_defined = "prior" in sas_func.keys()
            if is_prior_defined:
                self.transit_params["to_estimate"].append(name)
                self.params["to_estimate"][name] = sas_func["prior"]
                del sas_func["prior"]
            else:
                if is_C_old:
                    self.init_state_params.append(name)
                    self.params["to_estimate"][name] = {
                        "prior_dis": "normal",
                        "prior_params": [
                            sas_func[param_key],
                            sas_func[param_key] / 5.0,
                        ],
                        "is_nonnegative": True,
                    }
                else:
                    self.transit_params["to_estimate"].append(name)
                    self.params["to_estimate"][name] = {
                        "prior_dis": "normal",
                        "prior_params": [
                            sas_func["args"][param_key],
                            sas_func["args"][param_key] / 5.0,
                        ],
                        "is_nonnegative": True,
                    }

    def process_distribution_params(self, flux, sas_name, sas_func):
        dist = sas_func["func"]

        if dist == "kumaraswamy":
            if isinstance(sas_func["args"]["scale"], str):
                self.setup_prior_params(
                    "not_to_estimate", flux, sas_name, "scale", sas_func
                )
            else:
                self.setup_prior_params(
                    "to_estimate", flux, sas_name, "scale", sas_func
                )

        elif dist == "gamma":
            if isinstance(sas_func["args"]["scale"], str):
                if sas_func["args"]["a"] == 1.0:
                    self.setup_prior_params(
                        "not_to_estimate", flux, sas_name, "scale", sas_func
                    )
                    self.setup_prior_params(
                        "not_to_estimate", flux, sas_name, "a", sas_func
                    )
                else:
                    self.setup_prior_params(
                        "not_to_estimate", flux, sas_name, "scale", sas_func
                    )
                    self.setup_prior_params(
                        "to_estimate", flux, sas_name, "a", sas_func
                    )

            elif sas_func["args"]["a"] == 1.0:
                self.setup_prior_params(
                    "not_to_estimate", flux, sas_name, "a", sas_func
                )
                self.setup_prior_params(
                    "to_estimate", flux, sas_name, "scale", sas_func
                )

            else:
                self.setup_prior_params("to_estimate", flux, sas_name, "a", sas_func)
                self.setup_prior_params(
                    "to_estimate", flux, sas_name, "scale", sas_func
                )

    def set_obs_uncertainty(self, obs_uncertainty):
        for key, values in obs_uncertainty.items():
            self.params["to_estimate"][key] = values


# %%
class ModelInterfaceMesas:
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
        customized_model: SAS_Model,
        theta_init: Optional[dict] = None,
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
        self.N = num_input_scenarios
        self.T = len(self.df)  # set T to be the length of the dataframe
        self.R_prime = None  # preprocessed input scenarios
        self.num_states, self.num_obs = None, None
        self._start_ind, self._end_ind = None, None  # to track current time step

        # Set configurations according your own need here
        self.config = config
        self._parse_config()

        # Set initial values of parameters
        self._theta_init = None
        self._parse_theta_init(theta_init=theta_init)

        # initialize sas_model, given flux fixed, only need to initialize once
        self.model = customized_model(
            self.df,
            config={
                "sas_specs": self.sas_specs,
                "solute_parameters": self.solute_parameters,
                "options": self.options,
            },
            verbose=False,
        )

        # initialize theta
        self.update_theta()

    def _parse_config(self) -> None:
        """Parse config and set default values

        Set:
            dt (float): time step
            config (dict): configurations of the model
        """

        _default_config = {
            "dt": 1.0,
            "influx": ["J"],
            "outflux": ["Q", "ET"],
            "observed_made_each_step": True,
            "use_MAP_ref_traj": False,
            "use_MAP_AS_weight": False,
            "use_MAP_MCMC": False,
            "update_theta_dist": False,
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
                self.K = int(self.T // obs_made)
            else:
                self.K = int(self.T // obs_made) + 1
            self.observed_ind = np.arange(self.T, step=obs_made)

        else:
            raise ValueError("Error: Input format not supported!")

    def _set_fluxes(self) -> None:
        """Set influx and outflux based on config

        Set:
            influx (np.ndarray): influx data
            outflux (np.ndarray): outflux data
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

    def _parse_theta_init(self, theta_init: Optional[dict] = None) -> None:
        self.sas_specs = theta_init["sas_specs"]
        self.solute_parameters = theta_init["solute_parameters"]
        self.options = theta_init["options"]
        self.obs_uncertainty = theta_init["obs_uncertainty"]
        self._distinct_str = "#@#"

        # set _theta_init from sas_specs
        params_processor = ParamsProcessor(self._distinct_str)
        for flux, flux_sas in self.sas_specs.items():
            for sas_name, sas_func in flux_sas.items():
                params_processor.process_distribution_params(flux, sas_name, sas_func)

        # set _theta_init from solute_parameters
        self.conc_pairs = {}
        for in_conc, in_conc_params in self.solute_parameters.items():
            C_out = in_conc_params["observations"]

            if in_conc not in self.conc_pairs.keys():
                self.conc_pairs[in_conc] = [C_out]
            else:
                self.conc_pairs[in_conc].append(C_out)

            params_processor.setup_prior_params(
                "to_estimate", in_conc, C_out, "C_old", in_conc_params
            )

        self.in_sol = list(self.conc_pairs.keys())

        # set _theta_init from concentration pairs
        params_processor.set_obs_uncertainty(self.obs_uncertainty)

        self._theta_init = params_processor.params
        self._transit_params = params_processor.transit_params
        self._init_state_params = params_processor.init_state_params

        self._theta_to_estimate = list(self._theta_init["to_estimate"].keys())
        self._num_theta_to_estimate = len(self._theta_to_estimate)

        self.num_states = len(self.in_sol)
        self.num_obs = 1  # only C_Q is observed

        # Set parameter constraints and distributions
        self._set_parameter_constraints()
        self._set_parameter_distribution()

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
                    if std < mean / 10.0:
                        std = mean / 10.0

                # truncate or not
                if is_nonnegative:
                    a = (10.0e-6 - mean) / std
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

        # need to unpack here to get sas params

        # transition model
        transition_param = []
        for param_key in self._transit_params["to_estimate"]:
            transition_param.append(
                self._theta_init["to_estimate"][param_key]["current_value"]
            )
        for param_key in self._transit_params["not_to_estimate"]:
            transition_param.append(self._theta_init["not_to_estimate"][param_key])

        # observation model
        obs_param = [
            self._theta_init["to_estimate"]["sigma C out"]["current_value"],
        ]

        # input uncertainty param is to estimate
        input_param = [
            self._theta_init["to_estimate"]["sigma observed C in"]["current_value"],
            self._theta_init["to_estimate"]["sigma filled C in"]["current_value"],
        ]

        # initial state param is to estimate
        init_state = []
        for param_key in self._init_state_params:
            init_state.append(
                self._theta_init["to_estimate"][param_key]["current_value"]
            )

        # update model object theta
        self.theta = Parameter(
            input_model=input_param,
            transition_model=transition_param,
            observation_model=obs_param,
            initial_state=init_state,
        )

        # update bulk input and sas model every time theta is updated
        self._bulk_input_preprocess()
        self._init_sas_model()

        return

    def _bulk_input_preprocess(self) -> np.ndarray:
        """Preprocess input data

        U_obs: observed input concentration, C_in
        U_forcing: input forcing, J

        U_obs_true * U_forcing = (U_obs_observed + noise) * U_forcing

        Returns:
            np.ndarray: input data
        """

        is_input_obs = self.df["is_obs_input"].to_numpy()
        is_filled = self.df["is_obs_input_filled"].to_numpy()

        ipt_observed_ind_start = np.arange(self.T)[is_input_obs == True][:-1] + 1
        ipt_observed_ind_start = np.insert(ipt_observed_ind_start, 0, 0)
        ipt_observed_ind_end = np.arange(self.T)[is_input_obs == True] + 1

        input_obs = self.df[self.in_sol].to_numpy()
        input_forcing = self.influx.to_numpy()

        sig_r = input_obs.std(
            ddof=1
        )  # This is the input uncertainty to generate prediction scenarios

        # Bulk case: generate input scenarios based on observed input values

        self.R_prime = np.zeros((self.N, self.T))
        for i in range(sum(is_input_obs)):
            start_ind = ipt_observed_ind_start[i]
            end_ind = ipt_observed_ind_end[i]

            U_obs = input_obs[start_ind:end_ind]
            U_forcing = input_forcing[start_ind:end_ind]

            # different input uncertainty for filled and observed
            if is_filled[i]:
                sig_u = self.theta.input_model[1]
            else:
                sig_u = self.theta.input_model[0]

            for n in range(self.N):
                # R fluctuation is based on input fluctuation
                a, b = (0.0 - U_obs) / sig_r, np.inf
                R_temp = ss.truncnorm(a, b).rvs()

                if isinstance(R_temp, float):
                    R_temp = np.array([R_temp])
                # re-sample if R_temp is negative
                for r in range(len(R_temp)):
                    while R_temp[r] <= 0:
                        R_temp[r] = ss.truncnorm(a[r], b).rvs()

                # now generate observation uncertainty
                a = (0.0 - U_obs[0]) / sig_u
                R_obs = -np.inf
                while R_obs <= 0:
                    R_obs = ss.truncnorm(a, b).rvs()

                U_prime = R_obs * U_forcing
                R_temp = R_temp * U_forcing

                R_temp = normalize_over_interval(R_temp, U_prime) / U_forcing
                R_temp = np.nan_to_num(R_temp, nan=0.0)

                self.R_prime[n, start_ind:end_ind] = R_temp.ravel()

        # get dimension right
        self.R_prime = self.R_prime[:, :, np.newaxis]

    def input_model(self, start_ind: int, end_ind: int) -> None:
        """Input model to unpack input forcing, observation and generate input scenarios

        Output R_prime is the input scenarios for C_in from start_ind to end_ind

        """

        R_prime = self.R_prime[:, start_ind:end_ind, :]

        # record start and end ind for transition model
        self._start_ind, self._end_ind = start_ind, end_ind

        return R_prime

    def _init_sas_model(self) -> None:
        # create a new variable to storage sas model
        self._sas_funcs = {}
        self._sol_factors = {}

        # pass parameters to sas specs
        len_to_estimate = len(self._transit_params["to_estimate"])
        for i in range(len_to_estimate):
            param_key = self._transit_params["to_estimate"][i]
            flux, sas_name, param_name = param_key.split(self._distinct_str)
            self.model.sas_specs[flux].components[sas_name]._spec["args"][
                param_name
            ] = self.theta.transition_model[i]

        for i in range(len(self._transit_params["not_to_estimate"])):
            param_key = self._transit_params["not_to_estimate"][i]
            flux, sas_name, param_name = param_key.split(self._distinct_str)
            self.model.sas_specs[flux].components[sas_name]._spec["args"][
                param_name
            ] = self.theta.transition_model[i + len_to_estimate]

        temp_df = deepcopy(self.df)

        # pass parameters to solute parameters
        for i in range(len(self._init_state_params)):
            param_key = self._init_state_params[i]
            sol_in, sol_out, sol_init = param_key.split(self._distinct_str)
            if self.model.solute_parameters[sol_in]["observations"] == sol_out:
                self.model.solute_parameters[sol_in][
                    "C_old"
                ] = self.theta.initial_state[i]

            temp_df[sol_in] = 1.0
            temp_sol_param = deepcopy(self.model.solute_parameters)
            temp_sol_param[sol_in][sol_init] = 1.0

        # Get SAS function according to flux and sas_name
        self.model.run()
        for flux in self.model.fluxorder:
            self._sas_funcs[flux] = self.model.get_pQ(flux)

        # Get solute factors for each solute
        temp_model = self.model.copy_without_results()
        temp_model._data_df = temp_df
        temp_model.run()

        for i in range(len(self._init_state_params)):
            param_key = self._init_state_params[i]
            sol_in, sol_out, sol_init = param_key.split(self._distinct_str)
            self._sol_factors[sol_in] = self._get_col_factor(temp_model.get_CT("C in"))
        
    
    def _get_col_factor(self, CT: np.ndarray) -> np.ndarray:

        CT = np.nan_to_num(CT, nan=1.0)
        # Create an averaged CT
        new_CT = np.ones_like(CT)
        new_CT[0] = (CT[0] + 1.0) / 2.0
        # Update the remaining elements
        for i in range(1, CT.shape[0]):
            for j in range(i, CT.shape[1]):
                new_CT[i, j] = (CT[i - 1, j - 1] + CT[i, j]) / 2.0
        # Return the factor
        return new_CT[:, 1:]
        

    def transition_model(
        self, Xtm1: np.ndarray, Rt: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """State estimaton model f_theta(Xt-1, Rt)
         This is where to call mesas model

        For mesas: state variables are:
             - age-ranked mass of solute in the reservoir (mT)
             - age-ranked storage of the reservoir (ST)

         Args:
             Xtm1 (np.ndarray): state X at t = k-1 -> tracks C_in at time t
             Rt (np.ndarray, Optional): input signal at t = k-1:k

         Returns:
             np.ndarray: state X at t
        """

        num_iter = Rt.shape[1]

        Xt = np.zeros((self.N, num_iter, self.num_states))

        # use input and output fluxes to get partial mesas model
        self._start_ind
        self._end_ind


        # Get SAS function according to flux and sas_name  
        for flux in self.model.fluxorder:  
            pQ = self._sas_funcs[flux]
            for i, sol in enumerate(self.in_sol):
                C_J = self.model.data_df[sol].to_numpy()
                C_Q = np.zeros(self.T)
                # C_old is the state from last time step
                C_old = Xtm1[:, i]

                for n in range(self.N):




                    for t in range(self.T):
                        # the maximum age is t
                        for T in range(t + 1):
                            # the entry time is ti
                            ti = t - T
                            C_Q[t] += C_J[ti] * pQ[T, t] * self._sol_factors[sol][T, t] * self.dt

                        C_Q[t] += C_old * (1 - pQ[: t + 1, t].sum() * self.dt)
                    # save SAS function for each solute at each flux
                    self._sas_funcs[flux][sol] = C_Q


        return Xt

    def transition_model_probability(self, X_1toT: np.ndarray) -> np.ndarray:
        return 1.0

    def observation_model(self, Xk: np.ndarray) -> np.ndarray:
        """Observation probability g_theta(Xt)


        Returns:
            np.ndarray: y_hat at time k
        """
        if self._end_ind is None:
            _start_ind = 0
            _end_ind = 1
        else:
            _start_ind = self._start_ind
            _end_ind = self._end_ind

        length = Xk.shape[1]

        y_hat = np.zeros((self.N, length))
        for n in range(self.N):
            y_hat[n, :] = (self.model.result["C_Q"])[_start_ind:_end_ind, 0]

        return y_hat.ravel()

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
        yk = yk["Q"]
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
        state = np.zeros((num, len(self._init_state_params)))
        for i, state_name in enumerate(self._init_state_params):
            state[:, i] = self.dist_model[state_name].rvs(num)

        return state

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
