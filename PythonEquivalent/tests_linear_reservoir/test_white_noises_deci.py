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

# num_input_scenarios = int(sys.argv[1])
# num_parameter_samples = int(sys.argv[2])
# len_parameter_MCMC = int(sys.argv[3])
# k = float(sys.argv[4])
# ipt_std = float(sys.argv[5])
# obs_mode = sys.argv[6]
# interval = [0, int(sys.argv[7])]
# uncertainty_mode = sys.argv[8]
# %%
num_input_scenarios = 10
num_parameter_samples = 5
len_parameter_MCMC = 5
k = 1.0
ipt_std = 1.0
obs_mode = "deci_2d"
interval = [0, 20]
uncertainty_mode = "input"

# %%
ipt_mean = 5.0
test_case = "WhiteNoise"
data_root = "/Users/esthersida/pMESAS"

stn_input = [5]

length = 3000

model_interface_class = ModelInterfaceDeci

observation_patterns = ["deci both", "deci input", "deci output"]

# %%
def set_config(influx_type: str, outflux_type: str, obs_made: Any, obs_pattern: str, model_interface_class: Any):
    if obs_pattern == "deci both" or obs_pattern == "deci output":
        config = {
            "observed_made_each_step": obs_made,
            "influx": influx_type,
            "outflux": outflux_type,
            "use_MAP_AS_weight": False,
            "use_MAP_ref_traj": False,
            "use_MAP_MCMC": False,
            "update_theta_dist": False,
        }
        if obs_pattern == "deci output":
            model_interface_class = ModelInterfaceDeciFineInput

            
    elif obs_pattern == "deci input":
        config = {
            "observed_made_each_step": True,
            "influx": influx_type,
            "outflux": outflux_type,
            "use_MAP_AS_weight": False,
            "use_MAP_ref_traj": False,
            "use_MAP_MCMC": False,
            "update_theta_dist": False,
        }
    else:
        raise ValueError("Invalid observation pattern.")
    return config, model_interface_class

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

        if uncertainty_mode == "input":
            # path name
            path_str = f"{data_root}/Results/TestLR/{test_case}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k}_mean_{ipt_mean}_std_{ipt_std}_length_{interval[1]-interval[0]}/{case_name}_uncertain_input/{obs_pattern}"
            if not os.path.exists(path_str):
                os.makedirs(path_str)

            sig_ipt_hat = df_obs["J_obs"].std(ddof=1) / stn_i
            input_uncertainty_prior = [sig_ipt_hat, sig_ipt_hat / 3.0 ] 
            sig_obs_hat = df_obs["Q_true"].std(ddof=1)
            obs_uncertainty_prior = [sig_obs_hat / 100.0, sig_obs_hat / 100.0 / 3.0]

            config, model_interface_class=set_config("J_obs", "Q_true", obs_made, obs_pattern, model_interface_class)
        
        elif uncertainty_mode == "output":
            
            path_str = f"{data_root}/Results/TestLR/{test_case}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k}_mean_{ipt_mean}_std_{ipt_std}_length_{interval[1]-interval[0]}/{case_name}_uncertain_output/{obs_pattern}"
            if not os.path.exists(path_str):
                os.makedirs(path_str)

            sig_ipt_hat = df_obs["J_true"].std(ddof=1) / stn_i
            input_uncertainty_prior = [sig_ipt_hat / 100.0, sig_ipt_hat / 100.0 / 3.0]
            sig_obs_hat = df_obs["Q_obs"].std(ddof=1)
            obs_uncertainty_prior = [sig_obs_hat, sig_obs_hat / 3.0]

            config, model_interface_class=set_config("J_true", "Q_obs", obs_made, obs_pattern, model_interface_class)
        
        elif uncertainty_mode == "both":

            path_str = f"{data_root}/Results/TestLR/{test_case}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k}_mean_{ipt_mean}_std_{ipt_std}_length_{interval[1]-interval[0]}/{case_name}_uncertain_both/{obs_pattern}"
            if not os.path.exists(path_str):
                os.makedirs(path_str)

            sig_ipt_hat = df_obs["J_obs"].std(ddof=1) / stn_i
            input_uncertainty_prior = [sig_ipt_hat, sig_ipt_hat / 3.0]
            sig_obs_hat = df_obs["Q_obs"].std(ddof=1)
            obs_uncertainty_prior = [sig_obs_hat, sig_obs_hat / 3.0]

            config, model_interface_class=set_config("J_obs", "Q_obs", obs_made, obs_pattern, model_interface_class)

        # Set prior parameters
        k_prior = [k, k / 3.0]
        initial_state_prior = [df_obs["Q_obs"][0], df_obs["Q_obs"][0] / 3.0 * np.sqrt(k)]
        
        # Save prior parameters and compile
        prior_record = pd.DataFrame(
            [k_prior, initial_state_prior, input_uncertainty_prior, obs_uncertainty_prior]
        )

        prior_record.columns = ["mean", "std"]
        prior_record.index = ["k", "initial_state", "input_uncertainty", "obs_uncertainty"]
        prior_record.to_csv(f"{path_str}/prior_parameters_{stn_i}.csv")

        run_parameter = [num_input_scenarios, num_parameter_samples, len_parameter_MCMC]

        # run model
        ipt, opt, df = run_with_given_settings(
            df_obs,
            config,
            run_parameter,
            path_str,
            prior_record,
            plot_preliminary=False,
            model_interface_class=model_interface_class,
        )
    #%%
    plt.figure()
    plt.plot(df["J_true"], "k", linewidth=10)
    plt.plot(ipt[1:, :].T, marker='.')
    plt.show()
    # 
    plt.figure()
    plt.plot(df["Q_true"], "k", linewidth=10)
    plt.plot(opt[1:, :].T, marker='.')

    plt.show()

# %%
