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
from model.model_interface import ModelInterface
from tests_linear_reservoir.other_model_interfaces import ModelInterfaceBulk, ModelInterfaceDeci

# %%
# model run settings
num_input_scenarios = 10
num_parameter_samples = 10
len_parameter_MCMC = 10
test_case = "WhiteNoise"
# stn_input = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
stn_input = [5]
interval = [0, 30]
length = 3000
k_model = 100
model_interface_class = ModelInterfaceDeci
data_root = "/Users/esthersida/pMESAS"
obs_mode = "deci_2d"

for stn_i in stn_input:
    df = pd.read_csv(f"{data_root}/Data/{test_case}/stn_{stn_i}_T_{length}_k_{k_model}.csv", index_col=0)

    case = get_different_input_scenarios(df, interval, plot=False, observation_mode=obs_mode)

    df_obs = case.df_obs
    obs_made = case.obs_made
    case_name = case.case_name

    k_prior = [k_model, 3]
    initial_state_prior = [df_obs["Q_obs"][0], 0.01]
    sig_ipt_hat = df_obs["J_obs"].std(ddof=1) / stn_i
    input_uncertainty_prior = [sig_ipt_hat, sig_ipt_hat / 3.0]
    sig_obs_hat = df_obs["Q_obs"].std(ddof=1)
    obs_uncertainty_prior = [sig_obs_hat/10., sig_obs_hat/10. / 3.0]


    config = {"observed_made_each_step": obs_made, "outflux": "Q_true", "use_MAP_AS_weight": False, "use_MAP_ref_traj": False}

    # Save prior parameters
    path_str = f"{data_root}/Results/TestLR/{test_case}/k_{k_model}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}/{case_name}"
    
    if not os.path.exists(path_str):
        os.makedirs(path_str)

    # Save prior parameters and compile
    prior_record = pd.DataFrame([k_prior, initial_state_prior, input_uncertainty_prior, obs_uncertainty_prior])
    prior_record.columns = ["mean", "std"]
    prior_record.index = ["k", "initial_state", "input_uncertainty", "obs_uncertainty"]
    prior_record.to_csv(f"{path_str}/prior_parameters_{stn_i}.csv")

    run_parameter = [num_input_scenarios, num_parameter_samples, len_parameter_MCMC]

    # run model
    run_with_given_settings(df_obs, config, run_parameter, path_str, prior_record, plot_preliminary=True, model_interface_class=model_interface_class)


    # %%
    k = np.loadtxt(f"{path_str}/k.csv")
    initial_state = np.loadtxt(f"{path_str}/initial_state.csv")
    input_uncertainty = np.loadtxt(f"{path_str}/input_uncertainty.csv")
    obs_uncertainty = np.loadtxt(f"{path_str}/obs_uncertainty.csv")
    input_scenarios = np.loadtxt(f"{path_str}/input_scenarios.csv")
    output_scenarios = np.loadtxt(f"{path_str}/output_scenarios.csv")

    threshold = 0
    plot_df = pd.DataFrame(
        {
            "k": k,
            "initial state": initial_state,
            "input uncertainty": input_uncertainty,
            "obs uncertainty": obs_uncertainty,
        }
    )

    initial_state_true = 0.05001100239495894
    input_uncertainty_true = 0.2 / stn_i
    obs_uncertainty_true = 0.0
    dt = 1.0 / 24 / 60 * 15

    sig_e = dt * 0.2 / stn_i
    phi = 1 - k_model * dt
    sig_q = np.sqrt(sig_e**2 / (1 - phi**2))

    true_params = [
        k_model,
        initial_state_true,
        input_uncertainty_true,
        obs_uncertainty_true,
        sig_q,
    ]

    prior_params = pd.read_csv(f"{path_str}/prior_parameters_{stn_i}.csv", index_col=0)

    # plot posterior
    g = plot_parameter_posterior(plot_df, true_params, prior_params, threshold)
    g.savefig(f"{path_str}/posterior.pdf")

    # plot trajectories
    estimation = {"input": input_scenarios, "output": output_scenarios}
    truth_df = pd.read_csv(f"{path_str}/df.csv", index_col=0)

    g = plot_scenarios(truth_df, estimation, threshold, stn_i, sig_q)
    g.savefig(f"{path_str}/scenarios.pdf")

    g = plot_scenarios(truth_df, estimation, threshold, stn_i, sig_q, line_mode=True)
    g.savefig(f"{path_str}/scenarios_line.pdf")

    RMSE_J = np.sqrt(np.mean((estimation["input"][threshold:,1:].mean(axis=0) - truth_df["J_true"][1:])**2))
    RMSE_Q = np.sqrt(np.mean((estimation["output"][threshold:,:].mean(axis=0) - truth_df["Q_true"][:])**2))
    print(f"RSME_J: {RMSE_J}, theoretical: {input_uncertainty_true}")
    print(f"RSME_Q: {RMSE_Q}, theoretical: {obs_uncertainty_true}")

    # calculate KL divergence
    true_q_mean = truth_df["Q_true"].to_numpy()
    true_q_std = sig_q
    obs_q_mean = output_scenarios[threshold:,].mean(axis=0)
    obs_q_std = output_scenarios[threshold:,].std(axis=0)

    KL = cal_KL(true_q_std, obs_q_std, true_q_mean, obs_q_mean)
    # plot KL divergence
    plt.figure()
    plt.plot(KL[:-1])
    plt.xlabel("Timestep")
    plt.ylabel("KL divergence")
    plt.show()
    plt.savefig(f"{path_str}/KL_divergence.pdf")

    # convergence check
    convergence_check_plot(plot_df, 100)


# %%