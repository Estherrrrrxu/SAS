# %%
import pandas as pd
import matplotlib.pyplot as plt
from mesas_cases import *


# %%
# generate base plot
data_root = "/Users/esthersida/pMESAS/mesas"
# Data from publicated paper
df = pd.read_csv(f"{data_root}/data.csv", index_col=1, parse_dates=True)
df.columns = ["timestep", "J", "C in", "Q", "ET", "C out", "S_scale"]

# Orginal data
check_file = pd.read_csv(
    f"{data_root}/weekly_rainfall.csv", index_col=5, parse_dates=True
)
# Find where C in is observed
df["is_obs_input"] = check_file["Cl mg/l"]
df["is_obs_input"] = df["is_obs_input"].notna()
df["is_obs_input_filled"] = False
# Replace randomly generated cover data
df["C in raw"] = df["C in"][df["is_obs_input"]]
df["C in raw"] = df["C in raw"].backfill()
df["C in raw"][df["J"] == 0.0] = 0.0
# Replace randomly sampled data with monthly mean
ill_vals = df[["C in", "J"]][df["C in"] != df["C in raw"]]
monthly_means = ill_vals["C in"].groupby([ill_vals.index.month]).mean()
ill_vals["C in"] = monthly_means[ill_vals.index.month].to_list()

# define continuous time interval
fake_obs = (ill_vals.index.to_series().diff().dt.days >= 6).to_list()[1:]
fake_obs.append(True)
ill_vals["is_obs"] = fake_obs

# deal with across month condition
ill_vals["group"] = ill_vals["is_obs"].cumsum()
ill_vals["group"][ill_vals["is_obs"] == True] -= 1
for i in range(ill_vals["group"].iloc[-1] + 1):
    if len((ill_vals["C in"][ill_vals["group"] == i]).unique()) > 1:
        ill_vals["C in"][ill_vals["group"] == i] = (
            ill_vals["C in"][ill_vals["group"] == i]
            * ill_vals["J"][ill_vals["group"] == i]
        ).sum() / ill_vals["J"][ill_vals["group"] == i].sum()


# replace the ill values under new condition
df["C in raw"][df["C in"] != df["C in raw"]] = ill_vals["C in"]
df["is_obs_input"][df["C in"] != df["C in raw"]] = ill_vals["is_obs"]
df["is_obs_input_filled"][df["C in"] != df["C in raw"]] = True

df = df.iloc[:600]


# %%
# Load data
# result_root = "/Users/esthersida/pMESAS/mesas/Results/rockfish"
result_root = "/Users/esthersida/pMESAS/mesas/Results"
# case_names = ["invariant_q_u_et_u",  "storage_q_g_et_u"]

case_names = ["invariant_q_g_et_u"]
for case_name in case_names:
    qscale = f"{result_root}/qscale_{case_name}.csv"
    etscale = f"{result_root}/etscale_{case_name}.csv"
    c_old = f"{result_root}/c_old_{case_name}.csv"
    sigma_observed = f"{result_root}/sigma_observed_{case_name}.csv"
    sigma_filled = f"{result_root}/sigma_filled_{case_name}.csv"
    try:
        sigma_output = f"{result_root}/sigma_output_{case_name}.csv"
    except:
        print(case_name)
    input_scenarios = f"{result_root}/input_scenarios_{case_name}.csv"
    output_scenarios = f"{result_root}/output_scenarios_{case_name}.csv"

    qscale = pd.read_csv(qscale, header=None)
    etscale = pd.read_csv(etscale, header=None)
    c_old = pd.read_csv(c_old, header=None)
    sigma_observed = pd.read_csv(sigma_observed, header=None)
    sigma_filled = pd.read_csv(sigma_filled, header=None)
    sigma_output = pd.read_csv(sigma_output, header=None)
    input_scenarios = pd.read_csv(input_scenarios, header=None)
    output_scenarios = pd.read_csv(output_scenarios, header=None)

    # make the plot
    valid_ind_out = output_scenarios.loc[:, (output_scenarios != 0).any(axis=0)].columns
    valid_ind_in = df['J'].values != 0
    burn_in = 0
    # 
    # clean up raw output
    t = df.index
    C_in_hat = input_scenarios.values.T
    obs = df[df["is_obs_input"] == True].index

    # 
    #
    import seaborn as sns

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax0_1 = ax[0].twinx()
    ax0_1.bar(df.index,df["J"], color="C0", alpha=0.3, label="Precipitation")
    ax0_1.set_ylim(100, 0)
    ax[0].plot(t[valid_ind_in], C_in_hat[valid_ind_in,burn_in:], '.',markersize = 1.,color="gray", alpha=0.1)
    ax[0].plot(df["C in raw"].iloc[valid_ind_in], "_", color = "red", markersize=1)
    # ax[0].plot(df.index[df['is_obs_input'] & (- df['is_obs_input_filled'])], df["C in"][df['is_obs_input'] & (- df['is_obs_input_filled'])], "x", label="Observed C in")
    # ax[0].plot(df.index[df['is_obs_input_filled']], df["C in"][df['is_obs_input_filled']], ".", label="Filled C in")
    ax[0].set_ylim(50, -5)
    ax[0].set_xlim(obs[0], obs[-1])


    ax1_1 = ax[1].twinx()
    ax1_1.plot(df.index, df["Q"], color="C0", alpha=0.3, label="Flow rate")
    ax[1].plot(df.iloc[valid_ind_out,burn_in:].index, output_scenarios.iloc[burn_in:,valid_ind_out].values.T,".", color = "gray", alpha = 0.01)
    ax[1].plot(df["C out"], "x", color="red", label = 'Data')

    ax[0].set_ylabel("Concentration (mg/l)", fontsize=16)
    ax[1].set_ylabel("Concentration (mg/l)", fontsize=16)
    ax[0].set_title("Input", fontsize=18)
    ax[1].set_title("Output", fontsize=18)
    ax0_1.set_ylabel("Precipitation (mm/day)", fontsize=16)
    ax1_1.set_ylabel("Discharge (mm/day)", fontsize=16)

    # Get existing legend handles and labels
    handles, labels = ax[0].get_legend_handles_labels()
    handles.extend([plt.Line2D([0], [0], color='C0', lw =2, alpha = 0.5)])
    labels.extend(['Flow rate'])
    handles.extend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=5)])
    labels.extend(['Prediction'])
    handles.extend([plt.Line2D([0], [0], color='red', lw =2)])
    labels.extend(['Data'])

    orders = [1,0,2]
    h = [handles[i-1] for i in orders]
    l = [labels[i-1] for i in orders]
    ax[0].legend(h, l, frameon=False, loc="lower left")

    # Get existing legend handles and labels
    handles, labels = ax[1].get_legend_handles_labels()
    handles.extend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=5)])
    labels.extend(['Prediction'])
    handles.extend([plt.Line2D([0], [0], color='C0', lw =2, alpha = 0.5)])
    labels.extend(['Flow rate'])

    h = [handles[i-1] for i in [0,1,2]]
    l = [labels[i-1] for i in [0,1,2]]

    ax[1].legend(h, l, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{result_root}/input_output_{case_name}.pdf")


# %%
case_name = case_names[3]
if case_name == "invariant_q_u_et_u":
    config = theta_invariant_q_u_et_u
elif case_name == "invariant_q_g_et_u":
    config = theta_invariant_q_g_et_u
elif case_name == "invariant_q_g_et_e":
    config = theta_invariant_q_g_et_e
elif case_name == "storage_q_g_et_u":
    config = theta_storage_q_g_et_u

sigma_filled_paper = obs_uncertainty["sigma filled C in"]["prior_params"]
sigma_observed_paper = obs_uncertainty["sigma observed C in"]["prior_params"]
sigma_output_paper = obs_uncertainty["sigma C out"]["prior_params"]


#%%
# make a normal distribution pdf plot
from scipy.stats import norm
import numpy as np

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

x = np.linspace(0., 1.5, 100)
loc, scale = sigma_filled_paper[0], sigma_filled_paper[1]
pdf = norm(loc, scale).pdf(x)
pdf[pdf<0] = 0.
ax[0].plot(x, pdf, label="Prior")
ax[0].set_title("Sigma filled C in")
ax[0].set_xlabel("Sigma filled C in")
ax[0].set_ylabel("Density")

loc, scale = sigma_observed_paper[0], sigma_observed_paper[1]


x = np.linspace(0., 0.2, 100)
pdf = norm(loc, scale).pdf(x)
pdf[pdf<0] = 0.
ax[1].plot(x, pdf, label="Prior")
ax[1].set_title("Sigma observed C in")
ax[1].set_xlabel("Sigma observed C in")
ax[1].set_ylabel("Density")

x = np.linspace(0, 2, 100)
loc, scale = sigma_output_paper[0], sigma_output_paper[1]
pdf = norm(loc, scale).pdf(x)
pdf[pdf<0] = 0.
ax[2].plot(x, pdf, label="Prior")
ax[2].set_title("Sigma C out")
ax[2].set_xlabel("Sigma C out")
ax[2].set_ylabel("Density")

ax[0].hist(sigma_filled.values, density=True, label="Posterior")
ax[1].hist(sigma_observed.values, density=True, label="Posterior")
ax[2].hist(sigma_output.values, density=True, label="Posterior")

ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{result_root}/sigma_posterior_{case_name}.pdf")
# %%
qscale_paper = config["sas_specs"]["Q"]["Q SAS function"]["args"]["scale"]
etscale_paper = config["sas_specs"]["ET"]["ET SAS function"]["args"]["scale"]
c_old_paper = solute_parameters["C in"]["C_old"]


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].hist(qscale.values, density=True, label="Posterior")
ax[0].axvline(qscale_paper, color="red", label="Prior")
ax[0].set_title("Q scale")

ax[1].hist(etscale.values, density=True, label="Posterior")
ax[1].axvline(etscale_paper, color="red", label="Prior")
ax[1].set_title("ET scale")

ax[2].hist(c_old.values, density=True, label="Posterior")
ax[2].axvline(c_old_paper, color="red", label="Prior")
ax[2].set_title("C old")

ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{result_root}/scale_posterior_{case_name}.pdf")




# %%
