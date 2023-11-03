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
from post_processing_utils import *
# %%
# root directory to search
root_folder_name = "/Users/esthersida/pMESAS/Results/TestLR/WhiteNoise"

# Use os.walk to traverse the directory and its subdirectories
subdirs = []
for root, dirs, files in os.walk(root_folder_name):
    for dir_name in dirs:
        subdirs.append(os.path.join(root,dir_name))
dir_names = []
for subdir in subdirs:
    dir_name = subdir.split("/")
    dir_names.append(dir_name[7:])

# %%
# for perfect case
deci2d_ipt, deci4d_ipt, deci7d_ipt = [], [], []
deci2d_opt, deci4d_opt, deci7d_opt = [], [], []
deci2d_both, deci4d_both, deci7d_both = [], [], []


for d_name in dir_names:
    if len(d_name) == 1:
        continue
    if d_name[1] == "Decimated every 2d_uncertain_input" and d_name[0] not in deci2d_ipt:
        deci2d_ipt.append(d_name[0])
    if d_name[1] == "Decimated every 4d_uncertain_input" and d_name[0] not in deci4d_ipt:
        deci4d_ipt.append(d_name[0])
    if d_name[1] == "Decimated every 7d_uncertain_input" and d_name[0] not in deci7d_ipt:
        deci7d_ipt.append(d_name[0])
    if d_name[1] == "Decimated every 2d_uncertain_output" and d_name[0] not in deci2d_opt:
        deci2d_opt.append(d_name[0])
    if d_name[1] == "Decimated every 4d_uncertain_output" and d_name[0] not in deci4d_opt:
        deci4d_opt.append(d_name[0])
    if d_name[1] == "Decimated every 7d_uncertain_output" and d_name[0] not in deci7d_opt:
        deci7d_opt.append(d_name[0])
    if d_name[1] == "Decimated every 2d_uncertain_both" and d_name[0] not in deci2d_both:
        deci2d_both.append(d_name[0])
    if d_name[1] == "Decimated every 4d_uncertain_both" and d_name[0] not in deci4d_both:
        deci4d_both.append(d_name[0])
    if d_name[1] == "Decimated every 7d_uncertain_both" and d_name[0] not in deci7d_both:
        deci7d_both.append(d_name[0])


if len(deci2d_ipt) != len(deci2d_opt):
    print("Error: input and output have different number of subdirectories", len(deci2d_ipt), len(deci2d_opt))
if len(deci4d_ipt) != len(deci4d_opt):
    print("Error: input and output have different number of subdirectories", len(deci4d_ipt), len(deci4d_opt))
if len(deci7d_ipt) != len(deci7d_opt):
    print("Error: input and output have different number of subdirectories", len(deci7d_ipt), len(deci7d_opt))
# %%
stn_ratios, ks, means, stds, length = [], [], [], [], []
Ns, Ds, Ls = [], [], []
for p in deci2d_ipt:
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
lengths = np.unique(length)
Ns = np.unique(Ns)
Ds = np.unique(Ds)
Ls = np.unique(Ls)
print("Signal to noise ratio levels are:", stn_ratios)
print("N levels are:", Ns)
print("D levels are:", Ds)
print("L levels are:", Ls)
print("k levels are:", ks)
print("mean levels are:", means)
print("input standard deviation are:", stds)
print("lengths are:", lengths)


# %%
from post_processing_utils import *

num_input_scenarios = Ns[0]
num_parameter_samples = Ds[0]
len_parameter_MCMC = Ls[0]
ipt_mean = means[0]
le = length[0]
dt = 1.0 
# %%
case_names = ["Decimated every 2d", "Decimated every 4d", "Decimated every 7d"]
obs_modes = ["deci output", "deci input", "deci both"]


# case_names = ["Bulk every 2d", "Bulk every 4d", "Bulk every 7d"]
# obs_modes = ["bulk output", "bulk input", "bulk both"]

def get_RMSEs(stn_i, k_true, ipt_std, threshold, root_folder_name, case_name, obs_mode):
    RMSE_J_total, RMSE_Q_total, model_run_time, RMSE_J_obs, RMSE_Q_obs = cal_RMSE(
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
        root_folder_name,
        obs_mode
    )
    data_df = {
        "input_RMSE_total": RMSE_J_total,
        "output_RMSE_total": RMSE_Q_total,
        "input_RMSE_obs": RMSE_J_obs,
        "output_RMSE_obs": RMSE_Q_obs,
        "stn_i": stn_i,
        "k_true": k_true,
        "ipt_std": ipt_std,
        "threshold": threshold,
        "model_run_time": model_run_time,
        "obs_mode": obs_mode,
        "case_name": case_name,
    }
    return data_df
# %%
# Initialize an empty list to store dictionaries of data
data_list_input = []
data_list_output = []
data_list_both = []
make_plots = False

for case_name in case_names:
    for obs_mode in obs_modes:
        # Iterate over the nested loops
        for ipt_std in stds:
            for stn_i in stn_ratios:
                for k_true in ks:
                    for threshold in [30]:
                        case_name_input = f"{case_name}_uncertain_input"
                        case_name_output = f"{case_name}_uncertain_output"
                        case_name_both = f"{case_name}_uncertain_both"
                        
                        data_input = get_RMSEs(
                            stn_i,
                            k_true,
                            ipt_std,
                            threshold,
                            root_folder_name,
                            case_name_input,
                            obs_mode
                        )
                        data_output = get_RMSEs(
                            stn_i,
                            k_true,
                            ipt_std,
                            threshold,
                            root_folder_name,
                            case_name_output,
                            obs_mode
                        )
                        data_both = get_RMSEs(
                            stn_i,
                            k_true,
                            ipt_std,
                            threshold,
                            root_folder_name,
                            case_name_both,
                            obs_mode
                        )
                        data_list_input.append(data_input)
                        data_list_output.append(data_output)
                        data_list_both.append(data_both)

                        if make_plots:
                            plot_each_scenarios(
                                root_folder_name,
                                stn_i,
                                num_input_scenarios,
                                num_parameter_samples,
                                len_parameter_MCMC,
                                ipt_mean,
                                ipt_std,
                                k_true,
                                le,
                                case_name_input,
                                threshold,
                                obs_mode
                            )
                            plot_each_scenarios(
                                root_folder_name,
                                stn_i,
                                num_input_scenarios,
                                num_parameter_samples,
                                len_parameter_MCMC,
                                ipt_mean,
                                ipt_std,
                                k_true,
                                le,
                                case_name_output,
                                threshold,
                                obs_mode
                            )
                            plot_each_scenarios(
                                root_folder_name,
                                stn_i,
                                num_input_scenarios,
                                num_parameter_samples,
                                len_parameter_MCMC,
                                ipt_mean,
                                ipt_std,
                                k_true,
                                le,
                                case_name_both,
                                threshold,
                                obs_mode
                            ) 

# %%
data_list_input_df = pd.DataFrame(data_list_input)
data_list_output_df = pd.DataFrame(data_list_output)
data_list_both_df = pd.DataFrame(data_list_both)
# %%
data_list = pd.concat([data_list_input_df, data_list_output_df, data_list_both_df])
data_list['Uncertainty'] = data_list['case_name'].str.split("_").str[2]
data_list['Decimation'] = data_list['case_name'].str.split("_").str[0]

# %%
data_list.fillna(0, inplace=True)

# %%
# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
subset = data_list[data_list["Uncertainty"] == "input"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_total"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[0, 0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_total"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[1, 0],
    palette="muted",
    marker="o",
)
subset = data_list[data_list["Uncertainty"] == "output"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_total"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[0, 1],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_total"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[1, 1],
    palette="muted",
    marker="o",
)
subset = data_list[data_list["Uncertainty"] == "both"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_total"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[0, 2],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_total"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[1, 2],
    palette="muted",
    marker="o",
)

ax[0, 0].set_xscale("log")
ax[1, 0].set_xscale("log")
ax[0, 1].set_xscale("log")
ax[1, 1].set_xscale("log")
ax[0, 2].set_xscale("log")
ax[1, 2].set_xscale("log")
ax[0, 0].set_yscale("log")
ax[1, 0].set_yscale("log")
ax[0, 1].set_yscale("log")
ax[1, 1].set_yscale("log")
ax[0, 2].set_yscale("log")
ax[1, 2].set_yscale("log")

ax[0, 0].set_xlabel("")
ax[0, 1].set_xlabel("")
ax[0, 2].set_xlabel("")
ax[0, 1].set_ylabel("")
ax[1, 1].set_ylabel("")
ax[0, 2].set_ylabel("")
ax[1, 2].set_ylabel("")

ax[0, 0].set_title("Uncertain input", fontsize=15)
ax[0, 1].set_title("Uncertain output", fontsize=15)
ax[0, 2].set_title("Uncertain both", fontsize=15)

ax[0, 0].set_ylabel("Input RMSE", fontsize=14)
ax[1, 0].set_ylabel("Output RMSE", fontsize=14)

ax[1, 0].set_xlabel("True k", fontsize=14)
ax[1, 1].set_xlabel("True k", fontsize=14)
ax[1, 2].set_xlabel("True k", fontsize=14)

ax[0, 1].legend(frameon=False, ncol = 2)
ax[0, 0].legend().remove()
ax[0, 2].legend().remove()
ax[1, 0].legend().remove()
ax[1, 1].legend().remove()
ax[1, 2].legend().remove()

# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
subset = data_list[data_list["Uncertainty"] == "input"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_obs"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[0, 0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_obs"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[1, 0],
    palette="muted",
    marker="o",
)
subset = data_list[data_list["Uncertainty"] == "output"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_obs"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[0, 1],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_obs"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[1, 1],
    palette="muted",
    marker="o",
)
subset = data_list[data_list["Uncertainty"] == "both"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_obs"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[0, 2],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_obs"],
    hue="Decimation",
    style="obs_mode",
    data=subset,
    ax=ax[1, 2],
    palette="muted",
    marker="o",
)

ax[0, 0].set_xscale("log")
ax[1, 0].set_xscale("log")
ax[0, 1].set_xscale("log")
ax[1, 1].set_xscale("log")
ax[0, 2].set_xscale("log")
ax[1, 2].set_xscale("log")
ax[0, 0].set_yscale("log")
ax[1, 0].set_yscale("log")
ax[0, 1].set_yscale("log")
ax[1, 1].set_yscale("log")
ax[0, 2].set_yscale("log")
ax[1, 2].set_yscale("log")

ax[0, 0].set_xlabel("")
ax[0, 1].set_xlabel("")
ax[0, 2].set_xlabel("")
ax[0, 1].set_ylabel("")
ax[1, 1].set_ylabel("")
ax[0, 2].set_ylabel("")
ax[1, 2].set_ylabel("")

ax[0, 0].set_title("Uncertain input", fontsize=15)
ax[0, 1].set_title("Uncertain output", fontsize=15)
ax[0, 2].set_title("Uncertain both", fontsize=15)

ax[0, 0].set_ylabel("Input RMSE", fontsize=14)
ax[1, 0].set_ylabel("Output RMSE", fontsize=14)

ax[1, 0].set_xlabel("True k", fontsize=14)
ax[1, 1].set_xlabel("True k", fontsize=14)
ax[1, 2].set_xlabel("True k", fontsize=14)

ax[0, 1].legend(frameon=False, ncol = 2)
ax[0, 0].legend().remove()
ax[0, 2].legend().remove()
ax[1, 0].legend().remove()
ax[1, 1].legend().remove()
ax[1, 2].legend().remove()
# %%
