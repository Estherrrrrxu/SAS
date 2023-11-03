# %%
from typing import List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from test_utils import *


# %%
# calculate RMSE
def cal_RMSE(
    num_input_scenarios: int,
    num_parameter_samples: int,
    len_parameter_MCMC: int,
    ipt_mean: float,
    ipt_std: float,
    stn_i: int,
    k_true: float,
    le: int,
    case_name: str,
    threshold: int,
    result_root: str,
    obs_mode: str = None,
) -> List[float]:
    # %%
    if obs_mode is None:
        path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}"
    else:
        path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}/{obs_mode}"

    model_run_times = []
    if not os.path.exists(path_str):
        return None, None, None, None, None


    for filename in os.listdir(path_str):
        if filename.startswith("k"):
            model_run_times.append(float(filename[2:-4]))

    # only one run for now
    if len(model_run_times) == 0:
        return None, None, None, None, None
    
    #%%

    model_run_time = model_run_times[-1]

    input_scenarios = np.loadtxt(f"{path_str}/input_scenarios_{model_run_time}.csv")
    output_scenarios = np.loadtxt(f"{path_str}/output_scenarios_{model_run_time}.csv")

    estimation = {"input": input_scenarios, "output": output_scenarios}
    truth_df = pd.read_csv(f"{path_str}/df.csv", index_col=0)

    obs_ind = np.where(truth_df["is_obs"][:] == True)[0]

    full_input_mean = estimation["input"][threshold:, 1 : obs_ind[-1] + 1].mean(axis=0)
    full_output_mean = estimation["output"][threshold:, : obs_ind[-1] + 1].mean(axis=0)
    full_input_truth = truth_df["J_true"].iloc[1 : obs_ind[-1] + 1]
    full_output_truth = truth_df["Q_true"].iloc[: obs_ind[-1] + 1]

    RMSE_J_total = np.sqrt(np.mean((full_input_mean - full_input_truth) ** 2))
    RMSE_Q_total = np.sqrt(np.mean((full_output_mean - full_output_truth) ** 2))

    RMSE_J_obs = np.sqrt(
        np.mean((full_input_mean[obs_ind[1:] - 1] - full_input_truth[obs_ind[1:]]) ** 2)
    )
    RMSE_Q_obs = np.sqrt(
        np.mean((full_output_mean[obs_ind] - full_output_truth[obs_ind]) ** 2)
    )

    return RMSE_J_total, RMSE_Q_total, model_run_time, RMSE_J_obs, RMSE_Q_obs


# %%
def plot_each_scenarios(
    result_root,
    stn_i,
    num_input_scenarios,
    num_parameter_samples,
    len_parameter_MCMC,
    ipt_mean,
    ipt_std,
    k_true,
    le,
    case_name,
    threshold,
    obs_mode=None,
):
    if obs_mode is None:
        path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}"
    else:
        path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}/{obs_mode}"

    
    model_run_times = []

    if not os.path.exists(path_str):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"no model run for {path_str}")
        print("returning None")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None, None, None, None, None

    for filename in os.listdir(path_str):
        if filename.startswith("k"):
            model_run_times.append(float(filename[2:-4]))
    if len(model_run_times) == 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"no model run for {path_str}")
        print("returning None")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None, None, None, None, None       
    #%%
    # only one run for now
    model_run_time = model_run_times[-1]

    input_scenarios = np.loadtxt(f"{path_str}/input_scenarios_{model_run_time}.csv")
    output_scenarios = np.loadtxt(f"{path_str}/output_scenarios_{model_run_time}.csv")

    estimation = {"input": input_scenarios, "output": output_scenarios}
    truth_df = pd.read_csv(f"{path_str}/df.csv", index_col=0)

    obs_ind = np.where(truth_df["is_obs"][:] == True)[0]
    cn = case_name.split("_")
    uncertain_type = cn[-1]

    sig_e = ipt_std
    phi = 1 - k_true
    sig_q = np.sqrt(sig_e**2 / (1 - phi**2)) * k_true
    input_uncertainty_true = sig_e / stn_i
    obs_uncertainty_true = sig_q / stn_i 
    initial_state_true = truth_df["Q_true"].iloc[0]

    # load data
    k = np.loadtxt(f"{path_str}/k_{model_run_time}.csv")
    initial_state = np.loadtxt(f"{path_str}/initial_state_{model_run_time}.csv")
    input_uncertainty = np.loadtxt(f"{path_str}/input_uncertainty_{model_run_time}.csv")
    obs_uncertainty = np.loadtxt(f"{path_str}/obs_uncertainty_{model_run_time}.csv")
    prior_params = pd.read_csv(f"{path_str}/prior_parameters_{stn_i}.csv", index_col=0)

    # make plot dataframe
    plot_df = pd.DataFrame(
        {
            "k": k,
            "initial state": initial_state,
            "input uncertainty": input_uncertainty,
            "obs uncertainty": obs_uncertainty,
        }
    )

    # unpack true parameters
    true_params = [
        k_true,
        initial_state_true,
        input_uncertainty_true,
        obs_uncertainty_true,
    ]

    # plot posterior
    g = plot_parameter_posterior(plot_df, true_params, prior_params, threshold)
    g.savefig(f"{path_str}/posterior.pdf")

    # plot trajectories
    estimation = {"input": input_scenarios, "output": output_scenarios}
    truth_df = pd.read_csv(f"{path_str}/df.csv", index_col=0)

    g = plot_scenarios(
        truth_df,
        estimation,
        threshold,
        input_uncertainty_true,
        obs_uncertainty_true,
        line_mode=False,
        uncertain_input=uncertain_type,
    )
    g.savefig(f"{path_str}/scenarios.pdf")

    g = plot_scenarios(
        truth_df,
        estimation,
        threshold,
        input_uncertainty_true,
        obs_uncertainty_true,
        line_mode=True,
        uncertain_input=uncertain_type,
    )
    g.savefig(f"{path_str}/scenarios_line.pdf")

    # convergence check
    convergence_check_plot(plot_df, 100)
    plt.savefig(f"{path_str}/convergence_check.pdf")

    return None


# %%
