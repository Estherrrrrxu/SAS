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
deci2d, deci4d, deci7d = [], [], []


for d_name in dir_names:
    if len(d_name) == 1:
        continue
    if d_name[1] == "Decimated every 2d" and d_name[0] not in deci2d:
        deci2d.append(d_name[0])
    if d_name[1] == "Decimated every 4d" and d_name[0] not in deci4d:
        deci4d.append(d_name[0])
    if d_name[1] == "Decimated every 7d" and d_name[0] not in deci7d:
        deci7d.append(d_name[0])
#%%
stn_ratios, ks, means, stds, length = [], [], [], [], []
Ns, Ds, Ls = [], [], []
for p in deci2d:
    pp = p.split("_")
    stn_ratios.append(int(pp[0]))
    Ns.append(int(pp[2]))
    Ds.append(int(pp[4]))
    Ls.append(int(pp[6]))
    ks.append(float(pp[8]))
    means.append(float(pp[10]))
    stds.append(float(pp[12]))
    length.append(int(pp[14]))
# %%
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
    obs_mode: Optional[str] = None,
        
):
    #

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
    

    RMSE_J_total = np.sqrt(np.mean((estimation["input"][threshold:,1:obs_ind[-2]].mean(axis=0) - truth_df["J_true"].iloc[1:obs_ind[-2]])**2))
    RMSE_Q_total = np.sqrt(np.mean((estimation["output"][threshold:,:obs_ind[-2]].mean(axis=0) - truth_df["Q_true"][:obs_ind[-2]])**2))

    RMSE_J_obs = np.sqrt(np.mean((estimation["input"][threshold:,1:obs_ind[-2]][:,obs_ind[:-2]].mean(axis=0) - truth_df["J_true"].values[1:obs_ind[-2]][obs_ind[:-2]])**2))
    RMSE_Q_obs = np.sqrt(np.mean((estimation["output"][threshold:,:obs_ind[-2]][:,obs_ind[:-2]].mean(axis=0) - truth_df["Q_true"].values[1:obs_ind[-2]][obs_ind[:-2]])**2))



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


    return RMSE_J_total, RMSE_Q_total, model_run_time, RMSE_J_obs, RMSE_Q_obs

# %%

num_input_scenarios = Ns[0]
num_parameter_samples = Ds[0]
len_parameter_MCMC = Ls[0]
ipt_mean = means[0]
le = length[0]
dt = 1.0 
#%%
case_name = "Decimated every 4d"
#%%
import pandas as pd

# Initialize an empty list to store dictionaries of data
data_list = []

# Iterate over the nested loops
for ipt_std in stds:
    for stn_i in stn_ratios:
        for k_true in ks:
            for threshold in [20]:
                for obs_mode in ['matching', 'fine input', 'fine output']:
                    RMSE_J, RMSE_Q, model_run_time, RMSE_J_obs, RMSE_Q_obs = cal_RMSE(
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
                        make_plot=False,
                        obs_mode=obs_mode,
                    )
                    data = {
                        "RMSE_J_total": RMSE_J,
                        "RMSE_Q_total": RMSE_Q,
                        "stn_i": stn_i,
                        "k_true": k_true,
                        "ipt_std": ipt_std,
                        "threshold": threshold,
                        "model_run_time": model_run_time,
                        "obs_mode": obs_mode,
                        "RMSE_J_obs": RMSE_J_obs,
                        "RMSE_Q_obs": RMSE_Q_obs,
                    }
                    data_list.append(data)

# Create a DataFrame from the list of dictionaries
data_list = pd.DataFrame(data_list)

data_list.to_csv(f"{result_root}/RMSE_{case_name}.csv")

# %%
subset = data_list[data_list['stn_i'] == 5]
subset.columns = [
    "Input RMSE",
    "Output RMSE",
    "Signal to Noise",
    "True k",
    "Input st.dev",
    "threshold",
    "model_run_time",
    "obs_mode",
    "Input RMSE obs",
    "Output RMSE obs",
]

subset["Input theoretical RMSE"] = subset["Input st.dev"]/subset["Signal to Noise"]
subset["Output theoretical RMSE"] = np.sqrt(subset["Input theoretical RMSE"]**2 / (1 - (1 - subset["True k"])**2))
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.lineplot(data=subset, x="True k", y=subset["Input RMSE"], hue="obs_mode", ax=ax[0], marker="o")
sns.lineplot(data=subset, x="True k", y=subset["Output RMSE"], hue="obs_mode", ax=ax[1], marker="o")
ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[0].set_ylabel("Input total RMSE" )
ax[1].set_ylabel("Output total RMSE")


# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.lineplot(data=subset, x="True k", y=subset["Input RMSE obs"], hue="obs_mode", ax=ax[0], marker="o")
sns.lineplot(data=subset, x="True k", y=subset["Output RMSE obs"], hue="obs_mode", ax=ax[1], marker="o")
ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[1].set_yscale("log")

# %%
# %%
