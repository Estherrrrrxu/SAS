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

# %%
root_folder_name = "/Users/esthersida/pMESAS/Results_with_new/TestLR/WhiteNoise"

# Use os.walk to traverse the directory and its subdirectories
subdirs = []
for root, dirs, files in os.walk(root_folder_name):
    for dir_name in dirs:
        subdirs.append(os.path.join(root, dir_name))
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
        if (
            d_name[1] == "Almost perfect data_uncertain_input"
            and d_name[0] not in perfect_uncertain_input
        ):
            perfect_uncertain_input.append(d_name[0])
        if (
            d_name[1] == "Almost perfect data_uncertain_output"
            and d_name[0] not in perfect_uncertain_output
        ):
            perfect_uncertain_output.append(d_name[0])
        if (
            d_name[1] == "Almost perfect data_uncertain_both"
            and d_name[0] not in perfect_uncertain_both
        ):
            perfect_uncertain_both.append(d_name[0])

if len(perfect_uncertain_input) != len(perfect_uncertain_output):
    print(
        "Error: input and output have different number of subdirectories",
        len(perfect_uncertain_input),
        len(perfect_uncertain_output),
    )
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
make_plot = False

# Iterate over the nested loops
for ipt_std in stds:
    for stn_i in stn_ratios:
        for k_true in ks:
            for threshold in [30]:
                # sig_e = ipt_std * k_true
                # phi = 1 - k_true
                # sig_q = np.sqrt(sig_e**2 / (1 - phi**2))
                # sig_input = sig_e / stn_i / k_true
                # sig_output = sig_q / stn_i * (sig_e / sig_q)

                case_name_input = case_name + "_uncertain_input"
                data_input = get_RMSEs(
                    stn_i, k_true, ipt_std, threshold, root_folder_name, case_name_input
                )
                data_list_uncertain_input.append(data_input)

                case_name_output = case_name + "_uncertain_output"
                data_output = get_RMSEs(
                    stn_i,
                    k_true,
                    ipt_std,
                    threshold,
                    root_folder_name,
                    case_name_output,
                )
                data_list_uncertain_output.append(data_output)

                case_name_both = case_name + "_uncertain_both"
                data_both = get_RMSEs(
                    stn_i, k_true, ipt_std, threshold, root_folder_name, case_name_both
                )
                data_list_uncertain_both.append(data_both)

                if make_plot:
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
                        obs_mode=None,
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
                        obs_mode=None,
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
                        obs_mode=None,
                    )


# %%
data_list_uncertain_input = pd.DataFrame(data_list_uncertain_input)
data_list_uncertain_output = pd.DataFrame(data_list_uncertain_output)
data_list_uncertain_both = pd.DataFrame(data_list_uncertain_both)

# %%
data_list_uncertain_input["Uncertainty"] = "Input"
data_list_uncertain_output["Uncertainty"] = "Output"
data_list_uncertain_both["Uncertainty"] = "Both"
# %%
data_list = pd.concat(
    [data_list_uncertain_input, data_list_uncertain_output, data_list_uncertain_both]
)
data_list.columns
# %%
sig_e = data_list["ipt_std"] * data_list["k_true"]
phi = 1 - data_list["k_true"]
sig_q = np.sqrt(sig_e**2 / (1 - phi**2))
data_list["theoretical_ipt"] = sig_e / data_list["stn_i"] / data_list["k_true"]
data_list["theoretical_opt"] = sig_q / data_list["stn_i"] * (sig_e / sig_q)

# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
subset = data_list[data_list["Uncertainty"] == "Input"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_total"],
    hue="stn_i",
    data=subset,
    ax=ax[0, 0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_obs"],
    hue="stn_i",
    data=subset,
    ax=ax[1, 0],
    palette="muted",
    marker="o",
)
subset = data_list[data_list["Uncertainty"] == "Output"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_total"],
    hue="stn_i",
    data=subset,
    ax=ax[0, 1],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_obs"],
    hue="stn_i",
    data=subset,
    ax=ax[1, 1],
    palette="muted",
    marker="o",
)
subset = data_list[data_list["Uncertainty"] == "Both"]
sns.lineplot(
    x="k_true",
    y=subset["input_RMSE_total"],
    hue="stn_i",
    data=subset,
    ax=ax[0, 2],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=subset["output_RMSE_obs"],
    hue="stn_i",
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

ax[1, 0].legend(title="Signal to noise ratio", frameon=False)
ax[0, 1].legend().remove()
ax[0, 2].legend().remove()
ax[0, 0].legend().remove()
ax[1, 1].legend().remove()
ax[1, 2].legend().remove()
if not os.path.exists(f"{root_folder_name}/Perfect_traj"):
    os.makedirs(f"{root_folder_name}/Perfect_traj")

fig.savefig(f"{root_folder_name}/Perfect_traj/RMSE_total.pdf")
# %%
data_list.fillna(0, inplace=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.lineplot(
    x="k_true",
    y=data_list["input_RMSE_total"] / data_list["theoretical_ipt"],
    hue="stn_i",
    style="Uncertainty",
    data=data_list,
    ax=ax[0],
    palette="muted",
    marker="o",
)
sns.lineplot(
    x="k_true",
    y=data_list["output_RMSE_obs"] / data_list["theoretical_opt"],
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

ax[0].set_xlabel("True k", fontsize=14)
ax[1].set_xlabel("True k", fontsize=14)
ax[0].set_ylabel("RMSE/theoretical RMSE", fontsize=14)
ax[1].set_ylabel("")

ax[0].set_title("Input", fontsize=15)
ax[1].set_title("Output", fontsize=15)

handles, labels = ax[0].get_legend_handles_labels()
labels[0] = "Signal to noise ratio"
labels[4] = "Uncertainty type"

ax[0].legend(
    handles=handles[:], labels=labels[:], frameon=False, title_fontsize=12, ncols=2, bbox_to_anchor=(1, 0.45)
)
ax[1].legend().remove()
fig.savefig(f"{root_folder_name}/Perfect_traj/RMSE_total_ratio.pdf")


# %%
def get_plot_info(
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
):
    path_str = f"{result_root}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k_true}_mean_{ipt_mean}_std_{ipt_std}_length_{le}/{case_name}"

    model_run_times = []
    for filename in os.listdir(path_str):
        if filename.startswith("k"):
            model_run_times.append(float(filename[2:-4]))

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

    # plot trajectories
    estimation = {"input": input_scenarios, "output": output_scenarios}
    truth_df = pd.read_csv(f"{path_str}/df.csv", index_col=0)

    real_start = truth_df["index"][truth_df["is_obs"]].iloc[1]
    real_end = truth_df["index"][truth_df["is_obs"]].iloc[-1] + 1
    img_start = truth_df["index"][truth_df["is_obs"]].iloc[0]

    return (
        estimation,
        truth_df,
        real_start,
        real_end,
        img_start,
        uncertain_type,
        ax,
        input_uncertainty_true,
        obs_uncertainty_true,
    )


# %%

case_names = [
    case_name + "_uncertain_input",
    case_name + "_uncertain_output",
    case_name + "_uncertain_both",
]
# %%
stn_i = 3
N = 50
D = 25
L = 100
k_true = 0.01
ipt_mean = 5.0
ipt_std = 1.0
le = 30


# %%
for stn_i in [1, 3, 5]:
    for k_true in [0.001, 0.01, 0.1, 1.]:
# for stn_i in [3]:
#     for k_true in [0.01]:
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        for i in range(len(case_names)):
            case_name = case_names[i]

            (
                estimation,
                truth_df,
                real_start,
                real_end,
                img_start,
                uncertain_type,
                ax,
                sig_input,
                sig_output,
            ) = get_plot_info(
                root_folder_name,
                stn_i,
                num_input_scenarios,
                num_parameter_samples,
                len_parameter_MCMC,
                ipt_mean,
                ipt_std,
                k_true,
                le,
                case_name,
            )

            cn = case_name.split("_")
            uncertain_input = cn[-1]
            line_mode = False
            start_ind = 30

            if uncertain_input == "input" or uncertain_input == "both":
                ax[0, i].fill_between(
                    truth_df["index"][real_start:real_end],
                    truth_df["J_true"][real_start:real_end] - 1.96 * sig_input,
                    truth_df["J_true"][real_start:real_end] + 1.96 * sig_input,
                    color="grey",
                    alpha=0.3,
                    label=r"95% theoretical uncertainty bounds",
                )
            else:
                ax[0, i].fill_between(
                    truth_df["index"][real_start:real_end],
                    truth_df["J_true"][real_start:real_end],
                    truth_df["J_true"][real_start:real_end],
                    color="grey",
                    alpha=0.3,
                    label=r"95% theoretical uncertainty bounds",
                )

            # scenarios
            if line_mode:
                alpha = 5.0 / estimation["input"][start_ind:,].shape[0]
                ax[0, i].plot(
                    truth_df["index"][real_start:real_end],
                    estimation["input"][start_ind:, real_start:real_end].T,
                    color="C9",
                    linewidth=1.5,
                    zorder=0,
                    alpha=alpha,
                )

            else:
                sns.boxplot(
                    estimation["input"][start_ind:, :real_end],
                    orient="v",
                    ax=ax[0, i],
                    order=truth_df["index"] - img_start,
                    color="C9",
                    linewidth=1.5,
                    fliersize=1.5,
                    zorder=0,
                    width=0.8,
                    fill=False,
                    whis=(2.5, 97.5),
                )
                ax[0, i].plot(
                    truth_df["index"][real_start:real_end],
                    estimation["input"][start_ind:, real_start:real_end].mean(axis=0),
                    color="C9",
                    linewidth=3,
                    zorder=0,
                    label=f"Ensemble mean",
                )

            # truth trajectory
            ax[0, i].plot(
                truth_df["index"][real_start:real_end],
                truth_df["J_true"][real_start:real_end],
                color="k",
                label="Truth",
                linewidth=0.8,
            )

            # observations
            if uncertain_input == "input" or uncertain_input == "both":
                ax[0, i].scatter(
                    truth_df["index"][truth_df["is_obs"]][1:real_end],
                    truth_df["J_obs"][truth_df["is_obs"]][1:real_end],
                    marker="+",
                    c="k",
                    s=100,
                    linewidth=2,
                    label="Observations",
                )
            else:
                ax[0, i].scatter(
                    truth_df["index"][truth_df["is_obs"]][1:real_end],
                    truth_df["J_true"][truth_df["is_obs"]][1:real_end],
                    marker="+",
                    c="k",
                    s=100,
                    linewidth=2,
                    label="Observations",
                )

            # ========================================
            # uncertainty bounds

            if uncertain_input == "output" or uncertain_input == "both":
                ax[1, i].fill_between(
                    truth_df["index"][real_start:real_end],
                    truth_df["Q_true"][real_start:real_end] - 1.96 * sig_output,
                    truth_df["Q_true"][real_start:real_end] + 1.96 * sig_output,
                    color="grey",
                    alpha=0.3,
                    label=r"95% theoretical uncertainty",
                )
            else:
                ax[1, i].fill_between(
                    truth_df["index"][real_start:real_end],
                    truth_df["Q_true"][real_start:real_end],
                    truth_df["Q_true"][real_start:real_end],
                    color="grey",
                    alpha=0.3,
                    label=r"95% theoretical uncertainty",
                )

            # scenarios
            if line_mode:
                ax[1, i].plot(
                    truth_df["index"][real_start:real_end],
                    estimation["output"][start_ind:, real_start:real_end].T,
                    color="C9",
                    linewidth=1.5,
                    zorder=0,
                    alpha=alpha,
                )
            else:
                sns.boxplot(
                    estimation["output"][start_ind:, real_start:real_end],
                    order=truth_df["index"] - real_start,
                    orient="v",
                    ax=ax[1, i],
                    color="C9",
                    linewidth=1.5,
                    fliersize=1.5,
                    zorder=0,
                    width=0.8,
                    fill=False,
                    whis=(2.5, 97.5),
                )
                ax[1, i].plot(
                    truth_df["index"][real_start:real_end],
                    estimation["output"][start_ind:, real_start:real_end].mean(axis=0),
                    color="C9",
                    linewidth=3,
                    zorder=0,
                    label=f"Ensemble mean",
                )

            # truth trajectory
            ax[1, i].plot(
                truth_df["index"][real_start:real_end],
                truth_df["Q_true"][real_start:real_end],
                color="k",
                label="Truth",
                linewidth=0.8,
            )

            # observations
            if uncertain_input == "input":
                ax[1, i].scatter(
                    truth_df["index"][truth_df["is_obs"]][1:real_end],
                    truth_df["Q_true"][truth_df["is_obs"]][1:real_end],
                    marker="+",
                    c="k",
                    s=100,
                    linewidth=2,
                    label="Observations",
                )
            else:
                ax[1, i].scatter(
                    truth_df["index"][truth_df["is_obs"]][1:real_end],
                    truth_df["Q_obs"][truth_df["is_obs"]][1:real_end],
                    marker="+",
                    c="k",
                    s=100,
                    linewidth=2,
                    label="Observations",
                )

            ax[0, i].set_ylim(
                [
                    min(truth_df["J_true"] - 3 * sig_input),
                    max(truth_df["J_true"]) + 3 * sig_input,
                ]
            )
            ax[1, i].set_ylim(
                [
                    min(truth_df["Q_true"] - 3 * sig_output),
                    max(truth_df["Q_true"] + 3 * sig_output),
                ]
            )

            ax[0, i].set_xlim([real_start - 0.2, real_end + 0.2])
            ax[1, i].set_xlim([real_start - 0.2, real_end + 0.2])

            if i == 0:
                ax[0, i].set_ylabel("Input signal", fontsize=16)
                ax[0, i].set_xticks([])
                ax[1, i].set_ylabel("Output signal", fontsize=16)
            else:
                ax[0, i].set_ylabel("", fontsize=16)
                ax[0, i].set_xticks([])
                ax[1, i].set_ylabel("", fontsize=16)

            ax[1, i].set_xlabel("Timestep", fontsize=16)
            ax[1, i].set_xticks(np.arange(real_start, real_end, 5))

            if i == 1:
                if line_mode:
                    (cyan_line,) = ax[1, i].plot([], [], "c-", label="Scenarios")
                    ax[1, i].legend(
                        frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.1)
                    )
                else:
                    ax[1, i].legend(
                        frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.1)
                    )
            else:
                ax[1, i].legend().remove()

            ax[0,1].set_title(
                f"Signal to noise ratio = {stn_i}, k = {k_true}",
                fontsize=20,
            )
        fig.subplots_adjust(wspace=0.15, hspace=0.1)
        fig.savefig(
            f"{root_folder_name}/Perfect_traj/{uncertain_input}_stn_{stn_i}_k_{k_true}.pdf")


# %%
