
# %%
import numpy as np
from dataclasses import dataclass
import scipy.stats as ss
import pandas as pd
from typing import Optional, Any, List
from mesas.sas.model import Model
from functions.utils import normalize_over_interval

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
        self.params = {'to_estimate': {}, 'not_to_estimate': {}}
        self.distinct_str = distinct_str

    def setup_prior_params(self, mode, flux, sas_name, param_key, sas_func):
        name = self.distinct_str.join([flux, sas_name, param_key])

        # if this parameter is not to be estimated
        if mode == 'not_to_estimate':
            self.params['not_to_estimate'][name] = {sas_func['args'][param_key]}
        elif mode == 'to_estimate':
            is_C_old = param_key == 'C_old'
            is_prior_defined = 'prior' in sas_func.keys()
            if is_prior_defined:
                self.params['to_estimate'][name] = sas_func['prior']
                del sas_func['prior']
            else:
                if is_C_old:
                    self.params['to_estimate'][name] = {
                        'prior_dis': 'normal',
                        'prior_params': [sas_func[param_key], sas_func[param_key] / 5.],
                        'is_nonnegative': True,
                    }
                else:
                    self.params['to_estimate'][name] = {
                        'prior_dis': 'normal',
                        'prior_params': [sas_func['args'][param_key], sas_func['args'][param_key] / 5.],
                        'is_nonnegative': True,
                    }

    def process_distribution_params(self, flux, sas_name, sas_func):
        dist = sas_func['func']

        if dist == 'kumaraswamy':
            if isinstance(sas_func['args']['scale'], str):
                self.setup_prior_params('not_to_estimate', flux, sas_name, 'scale', sas_func)
            else:
                self.setup_prior_params('to_estimate', flux, sas_name, 'scale', sas_func)

        elif dist == 'gamma':

            if isinstance(sas_func['args']['scale'], str):
                if sas_func['args']['a'] == 1.:
                    self.setup_prior_params('not_to_estimate', flux, sas_name, 'scale', sas_func)
                    self.setup_prior_params('not_to_estimate', flux, sas_name, 'a', sas_func)
                else:
                    self.setup_prior_params('not_to_estimate', flux, sas_name, 'scale', sas_func)
                    self.setup_prior_params('to_estimate', flux, sas_name, 'a', sas_func)

            elif sas_func['args']['a'] == 1.:
                self.setup_prior_params('not_to_estimate', flux, sas_name, 'a', sas_func)
                self.setup_prior_params('to_estimate', flux, sas_name, 'scale', sas_func)

            else:
                self.setup_prior_params('to_estimate', flux, sas_name, 'a', sas_func)
                self.setup_prior_params('to_estimate', flux, sas_name, 'scale', sas_func)

    def set_obs_uncertainty(self, obs_uncertainty):
        for key, values in obs_uncertainty.items():
            self.params['to_estimate'][key] = values
        
#%%
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
        customized_model: Optional[Any] = None,
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
        self.model = [customized_model for n in range(self.N)]  # pass your own model here
        self.T = len(self.df)  # set T to be the length of the dataframe

        # Set configurations according your own need here
        self.config = config
        self._parse_config()

        # Set initial values of parameters
        self._theta_init = None
        self._parse_theta_init(theta_init=theta_init)

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
            "inflow": ["J"],
            "outflow": ["Q", "ET"],
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
        self._set_flowes()

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

    def _set_flowes(self) -> None:
        """Set inflow and outflow based on config

        Set:
            inflow (np.ndarray): inflow data
            outflow (np.ndarray): outflow data
        """
        # TODO: flexible way to insert multiple influx and outflux
        if self.config["inflow"] is not None:
            self.inflow = self.df[self.config["inflow"]]
        else:
            print("Warning: No inflow is given!")

        if self.config["outflow"] is not None:
            self.outflow = self.df[self.config["outflow"]]
        else:
            print("Warning: No outflow is given!")

        return
    
    def _parse_theta_init(self, theta_init: Optional[dict] = None) -> None:
        self.sas_specs = theta_init['sas_specs']
        self.solute_parameters = theta_init['solute_parameters']
        self.options = theta_init['options']
        self.obs_uncertainty = theta_init['obs_uncertainty']
        self._distinct_str = '#@#'

        # set _theta_init from sas_specs
        params_processor = ParamsProcessor(self._distinct_str)
        for flux, flux_sas in self.sas_specs.items():
            for sas_name, sas_func in flux_sas.items():
                params_processor.process_distribution_params(flux, sas_name, sas_func)
        
        # set _theta_init from solute_parameters
        self.conc_pairs = {}
        for in_conc, in_conc_params in self.solute_parameters.items():
            
            C_out = in_conc_params['observations']
            
            if in_conc not in self.conc_pairs.keys():
                self.conc_pairs[in_conc] = [C_out]
            else:
                self.conc_pairs[in_conc].append(C_out)

            params_processor.setup_prior_params('to_estimate', in_conc, C_out, 'C_old', in_conc_params)


        # set _theta_init from concentration pairs
        params_processor.set_obs_uncertainty(self.obs_uncertainty)
        
        self._theta_init = params_processor.params

        self._theta_to_estimate = list(self._theta_init['to_estimate'].keys())
        self._num_theta_to_estimate = len(self._theta_to_estimate)

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
                    if std < mean/10.:
                        std = mean/10.         

                # truncate or not
                if is_nonnegative:
                    a = (10.e-6 - mean) / std 
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
    
    # TODO: return to this after defining the model

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
    
    # %%
    def _bulk_input_preprocess(self) -> np.ndarray:
        """Preprocess input data

        U_obs: observed input concentration, C_in
        U_forcing: input forcing, J

        U_obs_true * U_forcing = (U_obs_observed + noise) * U_forcing

        Returns:
            np.ndarray: input data
        """

        input_obs = self.df['is_obs_input'].to_numpy()

        ipt_observed_ind_start = np.arange(self.T)[input_obs == True][:-1] + 1
        ipt_observed_ind_start = np.insert(ipt_observed_ind_start, 0, 0)
        ipt_observed_ind_end = np.arange(self.T)[input_obs == True] +1

        input_obs = self.influx['C_in'].to_numpy()
        input_forcing = self.influx['J'].to_numpy()

        sig_r = input_obs.std(ddof=1)


        # Bulk case: generate input scenarios based on observed input values

        self.R_prime = np.zeros((self.N, self.T))
        for i in range(sum(input_obs)):
            start_ind = ipt_observed_ind_start[i]
            end_ind = ipt_observed_ind_end[i]

            U_obs = input_obs[start_ind:end_ind]
            U_forcing = input_forcing[start_ind:end_ind]

            sig_u = self.theta.input_model

            for n in range(self.N):
                # R fluctuation is based on input fluctuation
                R_temp = ss.norm(U_obs, scale=sig_r).rvs()
                if R_temp[R_temp >0].size == 0:
                    R_temp[R_temp <= 0] = 10**(-8)
                else:
                    R_temp[R_temp <= 0] = min(10**(-8), min(R_temp[R_temp > 0]))
                
                U_prime = (ss.norm(U_obs[0], sig_u).rvs()) * U_forcing
                R_temp = R_temp * U_forcing
                R_temp = normalize_over_interval(R_temp, U_prime)
                
                self.R_prime[n,start_ind:end_ind] = R_temp/U_forcing


    def input_model(self, start_ind: int, end_ind: int) -> None:
        """Input model to unpack input forcing, observation and generate input scenarios

        Output R_prime is the input scenarios for C_in from start_ind to end_ind

        """
        if start_ind == 0 and end_ind == 1:
            self._bulk_input_preprocess()

        R_prime = self.R_prime[:,start_ind:end_ind]

        return R_prime
    
    def initialize_measa_model(self, data: dict, config: dict, verbose: bool = False):
        """Need to initialize self.N mesas model
        """
        self.model = self.model(data, config, verbose=verbose)
        return

    def transition_model(
        self, Xtm1: np.ndarray, Rt: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """State estimaton model f_theta(Xt-1, Rt)
        This is where to call mesas model

       For mesas: state variables are:
       M_in, M_out, S_T
       


        Args:
            Xtm1 (np.ndarray): state X at t = k-1
            Rt (np.ndarray, Optional): input signal at t = k-1:k

        Returns:
            np.ndarray: state X at t
        """
        for n in self.N:
            all_fluxes = pd.DataFrame(Rt[n], columns = self.config['all_fluxes'])


            
        # Get parameters
        theta = self.theta.transition_model
        temp_data = pd.DataFrame({})        # Create the model
        model = Model(data,
                    config=config_invariant_q_u_et_u,
                    verbose=False
                    )

        # Run the model
        model.run()

        # Extract results
        data_df = model.data_df
        flux = model.fluxorder[0]


        return Xt[:, 1:]  # return w/out initial state

    def transition_model_probability(self, X_1toT: np.ndarray) -> np.ndarray:
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

        return self.dist_model["initial_state"].rvs(num)

    def state_as_probability(self, offset: np.ndarray, std: float):
        """State model probability p(xk'|xk)

        Args:
            offset (np.ndarray): offset of state model (xk - xk')
            sd (float): standard deviation of state model, set to be adaptive for now

        Returns:
            np.ndarray: probability of estimated state around ref trajectory
        """

        return ss.norm(0, std).logpdf(offset)



    def sas_model_chunk(self, data: dict, start_ind: int, end_ind: int) -> None:
        """Interface with SAS model corresponding to each time period

        Args:
            data (dict): data for each time period
        """
        self.model = self.model(data, self.options, self.sas_specs, self.solute_parameters)

# %%
