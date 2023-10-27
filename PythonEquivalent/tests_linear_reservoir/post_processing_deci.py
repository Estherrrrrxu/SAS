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
result_root = "/Users/esthersida/pMESAS/Rockfish_results"

# Use os.walk to traverse the directory and its subdirectories
subdirs = []
for root, dirs, files in os.walk(result_root):
    for dir_name in dirs:
        subdirs.append(os.path.join(root,dir_name))
dir_names = []
for subdir in subdirs:
    dir_name = subdir.split("/")
    dir_names.append(dir_name[5:])

# %%
# for perfect case
deci2d_ipt, deci4d_ipt, deci7d_ipt = [], [], []
deci2d_opt, deci4d_opt, deci7d_opt = [], [], []


for d_name in dir_names:
    if len(d_name) == 1:
        continue
    if d_name[1] == "Decimated every 2d" and d_name[0] not in deci2d_ipt:
        deci2d_ipt.append(d_name[0])
    if d_name[1] == "Decimated every 4d" and d_name[0] not in deci4d_ipt:
        deci4d_ipt.append(d_name[0])
    if d_name[1] == "Decimated every 7d" and d_name[0] not in deci7d_ipt:
        deci7d_ipt.append(d_name[0])
    if d_name[1] == "Decimated every 2d_output" and d_name[0] not in deci2d_opt:
        deci2d_opt.append(d_name[0])
    if d_name[1] == "Decimated every 4d_output" and d_name[0] not in deci4d_opt:
        deci4d_opt.append(d_name[0])
    if d_name[1] == "Decimated every 7d_output" and d_name[0] not in deci7d_opt:
        deci7d_opt.append(d_name[0])

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
length = np.unique(length)
Ns = np.unique(Ns)
Ds = np.unique(Ds)
Ls = np.unique(Ls)
print("Signal to noise ratio levels are:", stn_ratios)
print("True k values are:", ks)
print("input standard deviation are:", stds)


# %%
num_input_scenarios = Ns[0]
num_parameter_samples = Ds[0]
len_parameter_MCMC = Ls[0]
ipt_mean = means[0]
le = length[0]
dt = 1.0 
# %%
case_name = "Decimated every 7d"
# %%
for obs_mode in ["fine output", "fine input", "matching"]:

    # Initialize an empty list to store dictionaries of data
    data_list_input = []
    data_list_output = []

    # Iterate over the nested loops
    for ipt_std in stds:
        for stn_i in stn_ratios:
            for k_true in ks:
                for threshold in [20]:
                    RMSE_J, RMSE_Q, model_run_time, RMSE_J_obs, RMSE_Q_obs = cal_RMSE_deci(
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
                        uncertain_input=True,
                        obs_mode=obs_mode,
                    )
                    data_input = {
                        "RMSE_J": RMSE_J,
                        "RMSE_Q": RMSE_Q,
                        "stn_i": stn_i,
                        "k_true": k_true,
                        "ipt_std": ipt_std,
                        "threshold": threshold,
                        "model_run_time": model_run_time,
                        "obs_mode": obs_mode,
                        "RMSE_J_obs": RMSE_J_obs,
                        "RMSE_Q_obs": RMSE_Q_obs,
                    }
                    data_list_input.append(data_input)


                    RMSE_J, RMSE_Q, model_run_time, RMSE_J_obs, RMSE_Q_obs = cal_RMSE_deci(
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
                        uncertain_input=False,
                        obs_mode=obs_mode,
                    )
                    data_output = {
                        "RMSE_J": RMSE_J,
                        "RMSE_Q": RMSE_Q,
                        "stn_i": stn_i,
                        "k_true": k_true,
                        "ipt_std": ipt_std,
                        "threshold": threshold,
                        "model_run_time": model_run_time,
                        "obs_mode": obs_mode,
                        "RMSE_J_obs": RMSE_J_obs,
                        "RMSE_Q_obs": RMSE_Q_obs,
                    }
                    data_list_output.append(data_output)
    # Create a DataFrame from the list of dictionaries
    data_list_input = pd.DataFrame(data_list_input)
    data_list_output = pd.DataFrame(data_list_output)

    data_list_input.to_csv(f"{result_root}/RMSE_{case_name}_{obs_mode}_in.csv")
    data_list_output.to_csv(f"{result_root}/RMSE_{case_name}_{obs_mode}_out.csv")

# %%
df_ipt_matching = pd.read_csv(f"{result_root}/RMSE_{case_name}_matching_in.csv", index_col=0)
df_ipt_fine_input = pd.read_csv(f"{result_root}/RMSE_{case_name}_fine input_in.csv", index_col=0)
df_ipt_fine_output = pd.read_csv(f"{result_root}/RMSE_{case_name}_fine output_in.csv", index_col=0)
df_opt_matching = pd.read_csv(f"{result_root}/RMSE_{case_name}_matching_out.csv", index_col=0)
df_opt_fine_input = pd.read_csv(f"{result_root}/RMSE_{case_name}_fine input_out.csv", index_col=0)
df_opt_fine_output = pd.read_csv(f"{result_root}/RMSE_{case_name}_fine output_out.csv", index_col=0)

df_ipt = pd.concat([df_ipt_matching, df_ipt_fine_input, df_ipt_fine_output])
df_opt = pd.concat([df_opt_matching, df_opt_fine_input, df_opt_fine_output])

# %%
df_ipt.columns = [
    "Input RMSE",
    "Output RMSE",
    "Signal to Noise ratio",
    "True k",
    "Input st.dev",
    "threshold",
    "Model run time",
    "Observation pattern",
    "Input RMSE at observed time",
    "Output RMSE at observed time",
]
df_opt.columns = [
    "Input RMSE",
    "Output RMSE",
    "Signal to Noise ratio",
    "True k",
    "Output st.dev",
    "threshold",
    "Model run time",
    "Observation pattern",
    "Input RMSE at observed time",
    "Output RMSE at observed time",
]


# %%
df_ipt["Theoretical RMSE"] = df_ipt["Input st.dev"]/df_ipt["Signal to Noise ratio"]
df_opt["Theoretical RMSE"] = df_opt["Output st.dev"]/df_opt["Signal to Noise ratio"]

# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
sns.lineplot(
    x="True k",
    y=df_ipt["Input RMSE"] / df_ipt["Theoretical RMSE"],
    hue="Observation pattern",
    data=df_ipt,
    ax=ax[0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="True k",
    y=df_opt["Output RMSE"] / df_opt["Theoretical RMSE"],
    hue="Observation pattern",
    data=df_opt,
    ax=ax[1],
    palette="muted",
    marker="o",
)

ax[0].set_xlabel("")
ax[1].set_xlabel("True k", fontsize=14)
ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[0].set_ylabel("RMSE/theoretical RMSE", fontsize = 14)
ax[1].set_ylabel("RMSE/theoretical RMSE", fontsize = 14)
ax[0].set_title("Input", fontsize = 15)
ax[1].set_title("Output", fontsize = 15)
ax[0].legend(frameon=False, title="Observation pattern", fontsize=12, loc = "upper right")
ax[1].legend(frameon=False, title="Observation pattern", fontsize=12, loc = "upper right")
plt.rcParams['legend.title_fontsize'] = 'larger'
fig.suptitle("Varying observation pattern", fontsize = 15)
fig.tight_layout()



# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
sns.lineplot(
    x="True k",
    y=df_ipt["Input RMSE at observed time"] / df_ipt["Theoretical RMSE"],
    hue="Observation pattern",
    data=df_ipt,
    ax=ax[0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="True k",
    y=df_opt["Output RMSE at observed time"] / df_opt["Theoretical RMSE"],
    hue="Observation pattern",
    data=df_opt,
    ax=ax[1],
    palette="muted",
    marker="o",
)

ax[0].set_xlabel("")
ax[1].set_xlabel("True k", fontsize=14)
ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[0].set_ylabel("RMSE/theoretical RMSE", fontsize = 14)
ax[1].set_ylabel("RMSE/theoretical RMSE", fontsize = 14)
ax[0].set_title("Input", fontsize = 15)
ax[1].set_title("Output", fontsize = 15)
ax[0].legend(frameon=False, title="Observation pattern", fontsize=12, loc = "upper right")
ax[1].legend(frameon=False, title="Observation pattern", fontsize=12, loc = "upper right")
plt.rcParams['legend.title_fontsize'] = 'larger'
fig.suptitle("Varying observation pattern", fontsize = 15)
fig.tight_layout()


# %%
