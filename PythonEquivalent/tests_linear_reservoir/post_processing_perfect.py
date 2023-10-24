# %%
import os

current_path = os.getcwd()
if current_path[-22:] != "tests_linear_reservoir":
    os.chdir("tests_linear_reservoir")
    print("Current working directory changed to 'tests_linear_reservoir'.")
import sys

sys.path.append("../")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from test_utils import *

# %%
# root directory to search
result_root = "/Users/esthersida/pMESAS/Results/TestLR/WhiteNoise"

# Use os.walk to traverse the directory and its subdirectories
subdirs = []
for root, dirs, files in os.walk(result_root):
    for dir_name in dirs:
        subdirs.append(os.path.join(root,dir_name))
dir_names = subdirs[48:]
dir_names = [dir_name[51:].split("/") for dir_name in dir_names]

# %%
# for perfect case
perfect = []

for d_name in dir_names:
    if d_name[1] == "Almost perfect data" and d_name[0] not in perfect:
        perfect.append(d_name[0])

stn_ratios, ks, means, stds, length = [], [], [], [], []
Ns, Ds, Ls = [], [], []
for p in perfect:
    pp = p.split("_")
    stn_ratios.append(int(pp[0]))
    Ns.append(int(pp[2]))
    Ds.append(int(pp[4]))
    Ls.append(int(pp[6]))
    ks.append(float(pp[8]))
    means.append(float(pp[10]))
    stds.append(float(pp[12]))
    length.append(int(pp[14]))

stn_ratios = np.unique(stn_ratios)
ks = np.unique(ks)
means = np.unique(means)
stds = np.unique(stds)
length = np.unique(length)
Ns = np.unique(Ns)
Ds = np.unique(Ds)
Ls = np.unique(Ls)
print("Signal to noise ratio levels are:", stn_ratios)
print("True k values are:", ks)
print("input standard deviation are:", stds)

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
        
):
    path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}"

    model_run_times = []
    for filename in os.listdir(path_str):
        if filename.startswith("k"):
            model_run_times.append(float(filename[2:-4]))

    model_run_time = model_run_times[0]


    input_scenarios = np.loadtxt(f"{path_str}/input_scenarios_{model_run_time}.csv")
    output_scenarios = np.loadtxt(f"{path_str}/output_scenarios_{model_run_time}.csv")

    estimation = {"input": input_scenarios, "output": output_scenarios}
    truth_df = pd.read_csv(f"{path_str}/df.csv", index_col=0)
    RMSE_J = np.sqrt(np.mean((estimation["input"][threshold:,1:].mean(axis=0) - truth_df["J_true"][1:])**2))
    RMSE_Q = np.sqrt(np.mean((estimation["output"][threshold:,:].mean(axis=0) - truth_df["Q_true"][:])**2))

    if make_plot:
        # theoretical values
        input_uncertainty_true = ipt_std / stn_i
        sig_e = input_uncertainty_true * dt
        phi = 1 - k_true * dt
        obs_uncertainty_true = np.sqrt(sig_e**2 / (1 - phi**2))
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

        g = plot_scenarios(truth_df, estimation, threshold, stn_i,input_uncertainty_true,obs_uncertainty_true, line_mode=False)
        g.savefig(f"{path_str}/scenarios.pdf")

        g = plot_scenarios(truth_df, estimation, threshold, stn_i,input_uncertainty_true,obs_uncertainty_true, line_mode=True)
        g.savefig(f"{path_str}/scenarios_line.pdf")

        # convergence check
        convergence_check_plot(plot_df, 100)


    return RMSE_J, RMSE_Q, model_run_time

# %%

num_input_scenarios = Ns[0]
num_parameter_samples = Ds[0]
len_parameter_MCMC = Ls[0]
ipt_mean = means[0]
le = length[0]
dt = 1.0 
case_name = "Almost perfect data"

import pandas as pd

# Initialize an empty list to store dictionaries of data
data_list = []

# Iterate over the nested loops
for ipt_std in stds:
    for stn_i in stn_ratios:
        for k_true in ks:
            for threshold in [20, 50, 70, 90]:
                RMSE_J, RMSE_Q, model_run_time = cal_RMSE(
                    num_input_scenarios,
                    num_parameter_samples,
                    len_parameter_MCMC,
                    ipt_mean,
                    ipt_std,
                    stn_i,
                    k_true,
                    le,
                    case_name,
                    threshold,
                    result_root,
                )
                data = {
                    "RMSE_J": RMSE_J,
                    "RMSE_Q": RMSE_Q,
                    "stn_i": stn_i,
                    "k_true": k_true,
                    "ipt_std": ipt_std,
                    "threshold": threshold,
                    "model_run_time": model_run_time,
                }
                data_list.append(data)

# Create a DataFrame from the list of dictionaries
data_list = pd.DataFrame(data_list)

data_list.to_csv(f"{result_root}/RMSE_{case_name}.csv")

# %%
# %%

