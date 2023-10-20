# %%
import os

current_path = os.getcwd()
if current_path[-22:] != "tests_linear_reservoir":
    os.chdir("tests_linear_reservoir")
    print("Current working directory changed to 'tests_linear_reservoir'.")
import sys

sys.path.append("../")

from functions.get_dataset import get_different_input_scenarios
from tests_linear_reservoir.test_utils import *
import pandas as pd

from tests_linear_reservoir.other_model_interfaces import ModelInterfaceDeci, ModelInterfaceDeciFineInput

# %%
# model run settings

num_input_scenarios = int(sys.argv[1])
num_parameter_samples = int(sys.argv[2])
len_parameter_MCMC = int(sys.argv[3])
k = float(sys.argv[4])
ipt_std = float(sys.argv[5])
obs_mode = sys.argv[6]
interval = [0, int(sys.argv[7])]
# %%
# num_input_scenarios = 5
# num_parameter_samples = 5
# len_parameter_MCMC = 5
# k = 1.0
# ipt_std = 1.0
# obs_mode = "deci_2d"
# interval = [0, 20]

# %%
ipt_mean = 5.0
test_case = "WhiteNoise"
data_root = "/Users/esthersida/pMESAS"

stn_input = [1, 3, 5]

length = 3000

model_interface_class = ModelInterfaceDeci

observation_patterns = ["matching", "fine output", "fine input"]

# %%
for stn_i in stn_input:
    # Read data and chop data
    df = pd.read_csv(
        f"{data_root}/Data/{test_case}/stn_{stn_i}_T_{length}_k_{k}_mean_{ipt_mean}_std_{ipt_std}.csv",
        index_col=0,
    )
    df = df.iloc[interval[0] : interval[1], :]

    # get specific running case
    case = get_different_input_scenarios(
        df, interval, plot=False, observation_mode=obs_mode
    )

    df_obs = case.df_obs
    obs_made = case.obs_made
    case_name = case.case_name

    for obs_pattern in observation_patterns:
        # Create result path
        path_str = f"{data_root}/Results/TestLR/{test_case}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k}_mean_{ipt_mean}_std_{ipt_std}_length_{interval[1]-interval[0]}/{case_name}/{obs_pattern}"
        if not os.path.exists(path_str):
            os.makedirs(path_str)

        # Set prior parameters
        k_prior = [k, k / 3.0]
        initial_state_prior = [df_obs["Q_obs"][0], df_obs["Q_obs"][0] / 3.0]
        sig_ipt_hat = df_obs["J_obs"].std(ddof=1) / stn_i
        input_uncertainty_prior = [sig_ipt_hat, sig_ipt_hat / 3.0]
        sig_obs_hat = df_obs["Q_obs"].std(ddof=1)
        obs_uncertainty_prior = [sig_obs_hat / 100.0, sig_obs_hat / 100.0 / 3.0]
        # Observation is set to be very small because currently using Q_true as the observation 
        
        if obs_pattern == "matching" or obs_pattern == "fine input":
            config = {
                "observed_made_each_step": obs_made,
                "outflux": "Q_true",
                "use_MAP_AS_weight": False,
                "use_MAP_ref_traj": False,
                "use_MAP_MCMC": False,
                "update_theta_dist": False,
            }
            if obs_pattern == "fine input":
                model_interface_class = ModelInterfaceDeciFineInput
        elif obs_pattern == "fine output":
            config = {
                "observed_made_each_step": 1,
                "outflux": "Q_true",
                "use_MAP_AS_weight": False,
                "use_MAP_ref_traj": False,
                "use_MAP_MCMC": False,
                "update_theta_dist": False,
            }
        else:
            raise ValueError("Invalid observation pattern.")
        
        # Save prior parameters and compile
        prior_record = pd.DataFrame(
            [k_prior, initial_state_prior, input_uncertainty_prior, obs_uncertainty_prior]
        )

        prior_record.columns = ["mean", "std"]
        prior_record.index = ["k", "initial_state", "input_uncertainty", "obs_uncertainty"]
        prior_record.to_csv(f"{path_str}/prior_parameters_{stn_i}.csv")

        run_parameter = [num_input_scenarios, num_parameter_samples, len_parameter_MCMC]

        # run model
        run_with_given_settings(
            df_obs,
            config,
            run_parameter,
            path_str,
            prior_record,
            plot_preliminary=False,
            model_interface_class=model_interface_class,
        )


# %%
