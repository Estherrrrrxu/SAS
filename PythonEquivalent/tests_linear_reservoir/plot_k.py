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
root_folder_name = "/Users/esthersida/pMESAS/Rockfish_results/TestLR/WhiteNoise"

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


# Initialize an empty list to store dictionaries of data
data_list_uncertain_input = []
data_list_uncertain_output = []
data_list_uncertain_both = []

# Iterate over the nested loops
for ipt_std in stds:
    for stn_i in stn_ratios:
        for k_true in ks:
            for threshold in [20]:
                case_name_input = case_name + "_uncertain_input"
                RMSE_J, RMSE_Q, model_run_time = cal_RMSE(
                    num_input_scenarios,
                    num_parameter_samples,
                    len_parameter_MCMC,
                    ipt_mean,
                    ipt_std,
                    stn_i,
                    k_true,
                    le,
                    case_name_input,
                    threshold,
                    root_folder_name,
                    make_plot=True,
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
                data_list_uncertain_input.append(data_input)

                case_name_output = case_name + "_uncertain_output"
                RMSE_J, RMSE_Q, model_run_time = cal_RMSE(
                    num_input_scenarios,
                    num_parameter_samples,
                    len_parameter_MCMC,
                    ipt_mean,
                    ipt_std,
                    stn_i,
                    k_true,
                    le,
                    case_name_output,
                    threshold,
                    root_folder_name,
                    make_plot=True,
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
                data_list_uncertain_output.append(data_output)

                case_name_both = case_name + "_uncertain_both"
                RMSE_J, RMSE_Q, model_run_time = cal_RMSE(
                    num_input_scenarios,
                    num_parameter_samples,
                    len_parameter_MCMC,
                    ipt_mean,
                    ipt_std,
                    stn_i,
                    k_true,
                    le,
                    case_name_both,
                    threshold,
                    root_folder_name,
                    make_plot=True,
                )
                data_both = {
                    "RMSE_J": RMSE_J,
                    "RMSE_Q": RMSE_Q,
                    "stn_i": stn_i,
                    "k_true": k_true,
                    "ipt_std": ipt_std,
                    "threshold": threshold,
                    "model_run_time": model_run_time,
                }


# %%
