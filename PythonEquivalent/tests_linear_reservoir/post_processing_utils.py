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
    make_plot: bool = False,
    uncertain_input: bool = True, 
    dt: float = 1.0,    
) -> List[float]:
    # path to the directory
    if uncertain_input:
        path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}"
    else:
        path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}_output"


    model_run_times = []
    for filename in os.listdir(path_str):
        if filename.startswith("k"):
            model_run_times.append(float(filename[2:-4]))

    # only one run for now
    model_run_time = model_run_times[0]

    input_scenarios = np.loadtxt(f"{path_str}/input_scenarios_{model_run_time}.csv")
    output_scenarios = np.loadtxt(f"{path_str}/output_scenarios_{model_run_time}.csv")

    estimation = {"input": input_scenarios, "output": output_scenarios}
    truth_df = pd.read_csv(f"{path_str}/df.csv", index_col=0)

    RMSE_J = np.sqrt(np.mean((estimation["input"][threshold:,1:].mean(axis=0) - truth_df["J_true"][1:])**2))
    RMSE_Q = np.sqrt(np.mean((estimation["output"][threshold:,:].mean(axis=0) - truth_df["Q_true"][:])**2))

    if make_plot:
        # theoretical values
        if uncertain_input:
            input_uncertainty_true = ipt_std / stn_i
            sig_e = input_uncertainty_true * dt
            phi = 1 - k_true * dt
            obs_uncertainty_true = np.sqrt(sig_e**2 / (1 - phi**2))
            initial_state_true = ipt_mean/k_true
        else:
            obs_uncertainty_true = ipt_std / stn_i
            phi = 1 - k_true * dt
            input_uncertainty_true = np.sqrt(obs_uncertainty_true ** 2 * (1 - phi**2))
            initial_state_true = ipt_mean/k_true


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

        g = plot_scenarios(truth_df, estimation, threshold, input_uncertainty_true, obs_uncertainty_true, line_mode=False, uncertain_input=uncertain_input)
        g.savefig(f"{path_str}/scenarios.pdf")

        g = plot_scenarios(truth_df, estimation, threshold, input_uncertainty_true, obs_uncertainty_true, line_mode=True, uncertain_input=uncertain_input)
        g.savefig(f"{path_str}/scenarios_line.pdf")

        # convergence check
        convergence_check_plot(plot_df, 100)


    return RMSE_J, RMSE_Q, model_run_time


# %%
# calculate RMSE
def cal_RMSE_deci(
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
    make_plot: bool = False,
    obs_mode: Optional[str] = None,
    uncertain_input: bool = True, 
    dt: float = 1.0,      
):
    #
    if uncertain_input:
        case_name = f"{case_name}"
    else:
        case_name = f"{case_name}_output"

    if obs_mode is None:
        path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}"
    else:
        path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}/{obs_mode}"

    model_run_times = []
    for filename in os.listdir(path_str):
        if filename.startswith("k"):
            model_run_times.append(float(filename[2:-4]))

    model_run_time = model_run_times[0]


    input_scenarios = np.loadtxt(f"{path_str}/input_scenarios_{model_run_time}.csv")
    output_scenarios = np.loadtxt(f"{path_str}/output_scenarios_{model_run_time}.csv")

    estimation = {"input": input_scenarios, "output": output_scenarios}
    truth_df = pd.read_csv(f"{path_str}/df.csv", index_col=0)
    obs_ind = np.where(truth_df['is_obs'][1:] == True)[0]
    

    RMSE_J_total = np.sqrt(np.mean((estimation["input"][threshold:,1:obs_ind[-1]].mean(axis=0) - truth_df["J_true"].iloc[1:obs_ind[-1]])**2))
    RMSE_Q_total = np.sqrt(np.mean((estimation["output"][threshold:,:obs_ind[-1]].mean(axis=0) - truth_df["Q_true"][:obs_ind[-1]])**2))

    RMSE_J_obs = np.sqrt(np.mean((estimation["input"][threshold:,1:obs_ind[-1]][:,obs_ind[:-1]].mean(axis=0) - truth_df["J_true"].values[1:obs_ind[-1]][obs_ind[:-1]])**2))
    RMSE_Q_obs = np.sqrt(np.mean((estimation["output"][threshold:,:obs_ind[-1]][:,obs_ind[:-1]].mean(axis=0) - truth_df["Q_true"].values[1:obs_ind[-1]][obs_ind[:-1]])**2))



    if make_plot:
        # theoretical values
        if uncertain_input:
            input_uncertainty_true = ipt_std / stn_i
            sig_e = input_uncertainty_true * dt
            phi = 1 - k_true * dt
            obs_uncertainty_true = np.sqrt(sig_e**2 / (1 - phi**2))
            initial_state_true = ipt_mean/k_true
        else:
            obs_uncertainty_true = ipt_std / stn_i
            phi = 1 - k_true * dt
            input_uncertainty_true = np.sqrt(obs_uncertainty_true ** 2 * (1 - phi**2))
            initial_state_true = ipt_mean/k_true

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

        g = plot_scenarios(truth_df, estimation, threshold, input_uncertainty_true,obs_uncertainty_true, line_mode=False, uncertain_input=uncertain_input)
        g.savefig(f"{path_str}/scenarios.pdf")

        g = plot_scenarios(truth_df, estimation, threshold,input_uncertainty_true,obs_uncertainty_true, line_mode=True, uncertain_input=uncertain_input)
        g.savefig(f"{path_str}/scenarios_line.pdf")

        # convergence check
        convergence_check_plot(plot_df, 100)


    return RMSE_J_total, RMSE_Q_total, model_run_time, RMSE_J_obs, RMSE_Q_obs
# %%
