# %%
import pandas as pd
import matplotlib.pyplot as plt
from mesas_cases import *
from scipy.stats import norm
import numpy as np


# %%
# generate base plot
data_root = "/Users/esthersida/pMESAS/mesas"
# Data from publicated paper
df = pd.read_csv(f"{data_root}/data_preprocessed.csv", index_col=1, parse_dates=True)
df = df.iloc[2250:3380]
df


# %%
# Load data
# result_root = "/Users/esthersida/pMESAS/mesas/Results/rockfish"
result_root = "/Users/esthersida/pMESAS/mesas/Results"
case_names = ["invariant_q_u_et_u",  "storage_q_g_et_u"]

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
    burn_in = 10
    # 
    # clean up raw output
    t = df.datetime
    C_in_hat = input_scenarios.values.T
    obs = df[df["is_obs_input"] == True].index

    # =============================================================================
    # Plot input and output
    # =============================================================================

    fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax0_1 = ax[0].twinx()
    ax0_1.bar(t,df["J"], color="C0", alpha=0.3, label="Precipitation")
    ax0_1.set_ylim(100, 0)
    ax[0].plot(t[valid_ind_in], C_in_hat[valid_ind_in,burn_in:], '.',markersize = 1.,color="gray", alpha=0.1)
    ax[0].plot(t[valid_ind_in], C_in_hat[valid_ind_in,:].mean(axis=1), '.',markersize = 1., color = "black")
    ax[0].plot(t[valid_ind_in], df["C in"].iloc[valid_ind_in], "_", color = "red", markersize=1)
    # ax[0].plot(df.index[df['is_obs_input'] & (- df['is_obs_input_filled'])], df["C in"][df['is_obs_input'] & (- df['is_obs_input_filled'])], "x", label="Observed C in")
    # ax[0].plot(df.index[df['is_obs_input_filled']], df["C in"][df['is_obs_input_filled']], ".", label="Filled C in")
    ax[0].set_ylim(-0.5, 50)
    ax[0].set_xlim(t.iloc[0], t[obs[-1]])

    ax1_1 = ax[1].twinx()
    ax1_1.plot(t, df["Q"], color="C0", alpha=0.3, label="Flow rate")
    ax[1].plot(t.values[valid_ind_out], output_scenarios.iloc[burn_in:,valid_ind_out].values.T,".", color = "gray", alpha = 0.01)
    ax[1].plot(t.values[valid_ind_out], output_scenarios.iloc[burn_in:,valid_ind_out].mean(axis=0), linewidth=1., color = "black", label = 'Prediction mean')
    ax[1].plot(t, df["C out"], "x", color="red", label = 'Data')

    ax[0].set_ylabel("Concentration (mg/l)", fontsize=16)
    ax[1].set_ylabel("Concentration (mg/l)", fontsize=16)
    ax[0].set_title("Inflow - Precipitation", fontsize=18)
    ax[1].set_title("Outflow - Discharge", fontsize=18)
    ax[1].set_ylim(4, 16)
    ax0_1.set_ylabel("Precipitation (mm/day)", fontsize=16)
    ax1_1.set_ylabel("Discharge (mm/day)", fontsize=16)

    # Get existing legend handles and labels
    handles, labels = ax[0].get_legend_handles_labels()
    handles.extend([plt.Line2D([0], [0], color='C0', lw =2, alpha = 0.5)])
    labels.extend(['Flow rate'])
    handles.extend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=5)])
    labels.extend(['Prediction mean'])
    handles.extend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=5)])
    labels.extend(['Prediction ensemble'])
    handles.extend([plt.Line2D([0], [0], color='red', lw =2)])
    labels.extend(['Data'])


    ax[0].legend(handles, labels, loc="upper right")

    # Get existing legend handles and labels
    handles, labels = ax[1].get_legend_handles_labels()
    handles.extend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=5)])
    labels.extend(['Prediction ensemble'])
    handles.extend([plt.Line2D([0], [0], color='C0', lw =2, alpha = 0.5)])
    labels.extend(['Flow rate'])

    h = [handles[i-1] for i in [0,1,3,2]]
    l = [labels[i-1] for i in [0,1,3,2]]

    ax[1].legend(h, l, loc="upper right")
    ax[1].set_xticks(t[obs[::10]])
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(45)

    fig.tight_layout()
    fig.savefig(f"{result_root}/input_output_{case_name}.pdf")


    # =============================================================================
    # Parameter posterior uncertainty
    # =============================================================================

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



    # 
    # make a normal distribution pdf plot

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    ax[0,0].hist(sigma_filled.values[burn_in:], 10, density=True, label="Posterior")
    ax[0,1].hist(sigma_observed.values[burn_in:], 10, density=True, label="Posterior")
    ax[0,2].hist(sigma_output.values[burn_in:], 10, density=True, label="Posterior")

    x = np.linspace(sigma_filled_paper[0] - 3*sigma_filled_paper[1], sigma_filled_paper[0] + 3*sigma_filled_paper[1], 1000)
    loc, scale = sigma_filled_paper[0], sigma_filled_paper[1]
    pdf = norm(loc, scale).pdf(x)
    pdf[pdf<0] = 0.
    ax[0,0].plot(x, pdf, label="Prior", color='red')
    ax[0,0].set_title(r"Interpolated $\mathtt{C_{J}}$ uncertainty $\sigma_i$ (parameter $\theta_{r}$)", fontsize=14)
    ax[0,0].set_ylabel("Density", fontsize=14)

    loc, scale = sigma_observed_paper[0], sigma_observed_paper[1]

    x = np.linspace(sigma_observed_paper[0] - 3*sigma_observed_paper[1], sigma_observed_paper[0] + 3*sigma_observed_paper[1], 1000)
    pdf = norm(loc, scale).pdf(x)
    pdf[pdf<0] = 0.
    ax[0,1].plot(x, pdf, label="Prior", color='red')
    ax[0,1].set_title(r"Measured $\mathtt{C_{J}}$ uncertainty $\sigma_m$ (parameter $\theta_{r}$)", fontsize=14)


    x = np.linspace(sigma_output_paper[0] - 3*sigma_output_paper[1], sigma_output_paper[0] + 3*sigma_output_paper[1], 1000)
    loc, scale = sigma_output_paper[0], sigma_output_paper[1]
    pdf = norm(loc, scale).pdf(x)
    pdf[pdf<0] = 0.
    ax[0,2].plot(x, pdf, label="Prior", color='red')
    ax[0,2].set_title(r"Outflow uncertainty $\sigma_y$ (parameter $\theta_{y}$)", fontsize=14)

    ax[0,1].legend(frameon=False, fontsize=14)

    #
    # =============================================================================
    # Parameter posterior SAS
    # =============================================================================
    qscale_paper = config["sas_specs"]["Q"]["Q SAS function"]["args"]["scale"]
    etscale_paper = config["sas_specs"]["ET"]["ET SAS function"]["args"]["scale"]
    c_old_paper = solute_parameters["C in"]["C_old"]

    
    if isinstance(qscale_paper, str):
        qscale_paper = config["sas_specs"]["Q"]["Q SAS function"]["args"]["a"]
    

    x = np.linspace(7./10.*c_old_paper, 13./10.*c_old_paper, 1000)
    pdf = norm(c_old_paper, c_old_paper/10.).pdf(x)
    ax[1,0].hist(c_old.values[burn_in:],10, density=True, label="Posterior")
    ax[1,0].plot(x, pdf, label="Prior", color='red')
    ax[1,0].set_title(r"Initial condition $\mathcal{C}_{o}$ (parameter $\theta_{C}$)", fontsize=14)

    x = np.linspace(0.7*qscale_paper, 1.3*qscale_paper, 1000)
    pdf = norm(qscale_paper, qscale_paper/10.).pdf(x)
    ax[1,1].hist(qscale.values[burn_in:],10, density=True, label="Posterior")
    ax[1,1].plot(x, pdf, label="Prior", color='red')
    ax[1,1].set_title(r"SAS parameter $\mu_\mathtt{Q}$ (parameter $\theta_{x}$)", fontsize=14)


    x = np.linspace(0.7*etscale_paper, 1.3*etscale_paper, 1000)
    pdf = norm(etscale_paper, etscale_paper/10.).pdf(x)
    ax[1,2].hist(etscale.values[burn_in:],10, density=True, label="Posterior")
    ax[1,2].plot(x, pdf, label="Prior", color='red')
    ax[1,2].set_title(r"SAS parameter $\mu_\mathtt{ET}$ (parameter $\theta_{x}$)", fontsize=14)

    ax[1,0].set_ylabel("Density", fontsize=14)

    fig.tight_layout()
    fig.savefig(f"{result_root}/parameter_posterior_{case_name}.pdf")




# %%
