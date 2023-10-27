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
perfect_uncertain_input = []
perfect_uncertain_output = []

for d_name in dir_names:
    if len(d_name) > 1:
        if d_name[1] == "Almost perfect data" and d_name[0] not in perfect_uncertain_input:
            perfect_uncertain_input.append(d_name[0])
        if d_name[1] == "Almost perfect data_output" and d_name[0] not in perfect_uncertain_output:
            perfect_uncertain_output.append(d_name[0])

if len(perfect_uncertain_input) != len(perfect_uncertain_output):
    print("Error: input and output have different number of subdirectories", len(perfect_uncertain_input), len(perfect_uncertain_output))
# %%
stn_ratios, ks, means, stds, length = [], [], [], [], []
Ns, Ds, Ls = [], [], []
for p in perfect_uncertain_input:
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
num_input_scenarios = Ns[0]
num_parameter_samples = Ds[0]
len_parameter_MCMC = Ls[0]
ipt_mean = means[0]
le = length[0]
dt = 1.0 
case_name = "Almost perfect data"


# Initialize an empty list to store dictionaries of data
data_list_input = []
data_list_output = []

# Iterate over the nested loops
for ipt_std in stds:
    for stn_i in stn_ratios:
        for k_true in ks:
            for threshold in [20]:
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
                    make_plot=False,
                    uncertain_input=True,
                )
                data_input = {
                    "RMSE_J": RMSE_J,
                    "RMSE_Q": RMSE_Q,
                    "stn_i": stn_i,
                    "k_true": k_true,
                    "ipt_std": ipt_std,
                    "threshold": threshold,
                    "model_run_time": model_run_time,
                }
                data_list_input.append(data_input)


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
                    make_plot=False,
                    uncertain_input=False,
                )
                data_output = {
                    "RMSE_J": RMSE_J,
                    "RMSE_Q": RMSE_Q,
                    "stn_i": stn_i,
                    "k_true": k_true,
                    "ipt_std": ipt_std,
                    "threshold": threshold,
                    "model_run_time": model_run_time,
                }
                data_list_output.append(data_output)
# %%
# Create a DataFrame from the list of dictionaries
data_list_input = pd.DataFrame(data_list_input)
data_list_output = pd.DataFrame(data_list_output)

data_list_input.to_csv(f"{result_root}/RMSE_{case_name}_in.csv")
data_list_output.to_csv(f"{result_root}/RMSE_{case_name}_out.csv")

# %%
data_list_input = pd.read_csv(f"{result_root}/RMSE_{case_name}_in.csv", index_col=0)
data_list_output = pd.read_csv(f"{result_root}/RMSE_{case_name}_out.csv", index_col=0)
# %%
data_list_input.columns = [
    "Input RMSE",
    "Output RMSE",
    "Signal to Noise ratio",
    "True k",
    "Input st.dev",
    "MCMC sample threshold",
    "Model run time",
]

data_list_output.columns = [
    "Input RMSE",
    "Output RMSE",
    "Signal to Noise ratio",
    "True k",
    "Output st.dev",
    "MCMC sample threshold",
    "Model run time",
]

data_list_input["Theoretical RMSE"] = data_list_input["Input st.dev"]/data_list_input["Signal to Noise ratio"]
data_list_output["Theoretical RMSE"] = data_list_output["Output st.dev"]/data_list_output["Signal to Noise ratio"]
# %%
df_subset_ipt = data_list_input[data_list_input["Signal to Noise ratio"] == 1.]
df_subset_opt = data_list_output[data_list_output["Signal to Noise ratio"] == 1.]

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
sns.lineplot(
    x="True k",
    y=df_subset_ipt["Input RMSE"] / df_subset_ipt["Theoretical RMSE"],
    hue="Input st.dev",
    data=df_subset_ipt,
    ax=ax[0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="True k",
    y=df_subset_opt["Output RMSE"] / df_subset_opt["Theoretical RMSE"],
    hue="Output st.dev",
    data=df_subset_opt,
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
ax[0].legend(frameon=False, title="Input st.dev", fontsize=12, loc = "upper right")
ax[1].legend(frameon=False, title="Input st.dev", fontsize=12, loc = "upper right")
plt.rcParams['legend.title_fontsize'] = 'larger'
fig.suptitle("Varying input signal standard deviation", fontsize = 15)
fig.tight_layout()



# %%

df_subset_ipt = data_list_input[data_list_input["Input st.dev"] == 1.]
df_subset_opt = data_list_output[data_list_output["Output st.dev"] == 1.]

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
sns.lineplot(
    x="True k",
    y=df_subset_ipt["Input RMSE"] / df_subset_ipt["Theoretical RMSE"],
    hue="Signal to Noise ratio",
    data=df_subset_ipt,
    ax=ax[0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="True k",
    y=df_subset_opt["Output RMSE"] / df_subset_opt["Theoretical RMSE"],
    hue="Signal to Noise ratio",
    data=df_subset_opt,
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
ax[0].legend(frameon=False, title="Signal to Noise", fontsize=12, loc = "upper right")
ax[1].legend(frameon=False, title="Signal to Noise", fontsize=12, loc = "upper right")
plt.rcParams['legend.title_fontsize'] = 'larger'
fig.suptitle("Varying input signal standard deviation", fontsize = 15)
fig.tight_layout()
# %%
