# %%
current_path = os.getcwd()
if current_path[-22:] != "tests_linear_reservoir":
    os.chdir("tests_linear_reservoir")
    print("Current working directory changed to 'tests_linear_reservoir'.")
import sys

sys.path.append("../")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# %%
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
perfect_uncertain_input = []
perfect_uncertain_output = []
perfect_uncertain_both = []

for d_name in dir_names:
    if len(d_name) > 1:
        if d_name[1] == "Almost perfect data_uncertain_input" and d_name[0] not in perfect_uncertain_input:
            perfect_uncertain_input.append(d_name[0])
        if d_name[1] == "Almost perfect data_uncertain_output" and d_name[0] not in perfect_uncertain_output:
            perfect_uncertain_output.append(d_name[0])
        if d_name[1] == "Almost perfect data_uncertain_both" and d_name[0] not in perfect_uncertain_both:
            perfect_uncertain_both.append(d_name[0])

if len(perfect_uncertain_input) != len(perfect_uncertain_output):
    print("Error: input and output have different number of subdirectories", len(perfect_uncertain_input), len(perfect_uncertain_output))
# %%
stn_ratios, Ns, Ds, Ls, ks, means, stds, lengths = [], [], [], [], [], [], [], []
for p in perfect_uncertain_input:
    pp = p.split("_")
    stn_ratios.append(int(pp[0]))
    Ns.append(int(pp[2]))
    Ds.append(int(pp[4]))
    Ls.append(int(pp[6]))
    ks.append(float(pp[8]))
    means.append(float(pp[10]))
    stds.append(float(pp[12]))
    lengths.append(int(pp[14]))

stn_ratios = np.unique(stn_ratios)
ks = np.unique(ks)
means = np.unique(means)
stds = np.unique(stds)
lengths = np.unique(lengths)
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
le = lengths[0]
dt = 1.0 
case_name = "Almost perfect data"

# call function to get data
def get_RMSEs(stn_i, k_true, ipt_std, threshold, root_folder_name, case_name):
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
    }
    return data_df



# Initialize an empty list to store dictionaries of data
data_list_uncertain_input = []
data_list_uncertain_output = []
data_list_uncertain_both = []

# Iterate over the nested loops
for ipt_std in stds:
    for stn_i in stn_ratios:
        for k_true in ks:
            for threshold in [30]:

                case_name_input = case_name + "_uncertain_input"
                data_input = get_RMSEs(stn_i, k_true, ipt_std, threshold, root_folder_name, case_name_input)
                data_list_uncertain_input.append(data_input)

                case_name_output = case_name + "_uncertain_output"
                data_output = get_RMSEs(stn_i, k_true, ipt_std, threshold, root_folder_name, case_name_output)
                data_list_uncertain_output.append(data_output)

                case_name_both = case_name + "_uncertain_both"
                data_both = get_RMSEs(stn_i, k_true, ipt_std, threshold, root_folder_name, case_name_both)
                data_list_uncertain_both.append(data_both)

# %%
data_list_uncertain_input = pd.DataFrame(data_list_uncertain_input)
data_list_uncertain_output = pd.DataFrame(data_list_uncertain_output)
data_list_uncertain_both = pd.DataFrame(data_list_uncertain_both)
# %%
data_list_uncertain_input['Uncertainty'] = 'Input'
data_list_uncertain_output['Uncertainty'] = 'Output'
data_list_uncertain_both['Uncertainty'] = 'Both'
# %%
data_list = pd.concat([data_list_uncertain_input, data_list_uncertain_output, data_list_uncertain_both])
data_list.columns
# %%
sig_e = data_list['ipt_std'] * data_list['k_true']
phi = 1 - data_list['k_true']
sig_q = np.sqrt(sig_e**2 / (1 - phi**2))
data_list['theoretical_ipt'] = sig_e / data_list['stn_i'] / data_list['k_true']
data_list['theoretical_opt'] = sig_q / data_list['stn_i'] * (sig_e/sig_q)
    
# %%
fig, ax = plt.subplots(2, 3, figsize=(15,9))
subset = data_list[data_list["Uncertainty"] == "Input"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_total"],
    hue="stn_i",
    data=subset,
    ax=ax[0,0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_obs"],
    hue="stn_i",
    data=subset,
    ax=ax[1,0],
    palette="muted",
    marker="o",
)
subset = data_list[data_list["Uncertainty"] == "Output"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_total"],
    hue="stn_i",
    data=subset,
    ax=ax[0,1],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_obs"],
    hue="stn_i",
    data=subset,
    ax=ax[1,1],
    palette="muted",
    marker="o",
)
subset = data_list[data_list["Uncertainty"] == "Both"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_total"],
    hue="stn_i",
    data=subset,
    ax=ax[0,2],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_obs"],
    hue="stn_i",
    data=subset,
    ax=ax[1,2],
    palette="muted",
    marker="o",
)

ax[0,0].set_xscale("log")
ax[1,0].set_xscale("log")
ax[0,1].set_xscale("log")
ax[1,1].set_xscale("log")
ax[0,2].set_xscale("log")
ax[1,2].set_xscale("log")
ax[0,0].set_yscale("log")
ax[1,0].set_yscale("log")
ax[0,1].set_yscale("log")
ax[1,1].set_yscale("log")
ax[0,2].set_yscale("log")
ax[1,2].set_yscale("log")

ax[0,0].set_xlabel("")
ax[0,1].set_xlabel("")
ax[0,2].set_xlabel("")
ax[0,1].set_ylabel("")
ax[1,1].set_ylabel("")
ax[0,2].set_ylabel("")
ax[1,2].set_ylabel("")

ax[0,0].set_title("Uncertain input", fontsize = 15)
ax[0,1].set_title("Uncertain output", fontsize = 15)
ax[0,2].set_title("Uncertain both", fontsize = 15)

ax[0,0].set_ylabel("Input RMSE", fontsize = 14)
ax[1,0].set_ylabel("Output RMSE", fontsize = 14)

ax[1,0].set_xlabel("True k", fontsize = 14)
ax[1,1].set_xlabel("True k", fontsize = 14)
ax[1,2].set_xlabel("True k", fontsize = 14)

ax[0,0].legend(title="Signal to noise ratio", frameon=False)
ax[0,1].legend().remove()
ax[0,2].legend().remove()
ax[1,0].legend().remove()
ax[1,1].legend().remove()
ax[1,2].legend().remove()
    
# %%
fig, ax = plt.subplots(1, 2, figsize=(12,6))

sns.lineplot(
    x="k_true",
    y=data_list["input_RMSE_total"]/data_list["theoretical_ipt"],
    hue="stn_i",
    style="Uncertainty",
    data=data_list,
    ax=ax[0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=data_list["output_RMSE_obs"]/data_list["theoretical_opt"],
    hue="stn_i",
    style="Uncertainty",
    data=data_list,
    ax=ax[1],
    palette="muted",
    marker="o",
)

ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[0].set_yscale("log")
ax[1].set_yscale("log")

ax[0].set_xlabel("True k", fontsize = 14)
ax[1].set_xlabel("True k", fontsize = 14)
ax[0].set_ylabel("RMSE/theoretical RMSE", fontsize = 14)
ax[1].set_ylabel("")

ax[0].set_title("Input", fontsize = 15)
ax[1].set_title("Output", fontsize = 15)

handles, labels = ax[0].get_legend_handles_labels()
labels[0] = "Signal to noise ratio"
labels[4] = "Uncertainty type"

ax[0].legend(handles=handles[:], labels=labels[:], frameon=False, title_fontsize = 12, ncols=2)
ax[1].legend().remove()


# %%
