# %%
import pandas as pd
import matplotlib.pyplot as plt
from mesas_cases import *
from scipy.stats import norm
import numpy as np

from scipy.stats import gamma

# %%
# generate base plot
data_root = "/Users/esthersida/pMESAS/mesas"
# Data from publicated paper
df = pd.read_csv(f"{data_root}/data_preprocessed.csv", index_col=1, parse_dates=True)
df = df.iloc[2250:3383]
df

# %%
result_root = "/Users/esthersida/pMESAS/mesas/Results"
case_name = 'storage_q_g_et_u_changing_varying'
theta = f"{result_root}/theta_{case_name}.csv"
input_scenarios = f"{result_root}/input_scenarios_{case_name}.csv"
output_scenarios = f"{result_root}/output_scenarios_{case_name}.csv"
convolved = f"{result_root}/Results/C_Q_conv_storage_q_g_et_u.csv"

theta = pd.read_csv(theta)
input_scenarios = pd.read_csv(input_scenarios, header=None)
output_scenarios = pd.read_csv(output_scenarios, header=None)
convolved = pd.read_csv(convolved, header=None).values.flatten()

# %%
# =============================================================================
# 1. Plot input and output
# =============================================================================

# make the plot
valid_ind_out = output_scenarios.loc[:, (output_scenarios != 0).any(axis=0)].columns
valid_ind_in = df['J'].values != 0
burn_in = 10


# clean up raw output
t = df.datetime
C_in_hat = input_scenarios.values.T
obs = df[df["is_obs_input"] == True].index
color_data = "#D62728"
color_convol = "#FF9896"

# =============================================================================
# Plot input and output
# =============================================================================

fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
ax0_1 = ax[0].twinx()
ax0_1.bar(t,df["J"], color="C0", alpha=0.3, label="Precipitation")
ax0_1.set_ylim(100, 0)
ax[0].plot(t[valid_ind_in], C_in_hat[valid_ind_in,burn_in:], '.',markersize = 1.,color="gray", alpha=0.1)
ax[0].plot(t[valid_ind_in], C_in_hat[valid_ind_in,:].mean(axis=1), '.',markersize = 1., color = "black")
ax[0].plot(t[valid_ind_in], df["C in"].iloc[valid_ind_in], "_", color = color_data, markersize=1)
# ax[0].plot(df.index[df['is_obs_input'] & (- df['is_obs_input_filled'])], df["C in"][df['is_obs_input'] & (- df['is_obs_input_filled'])], "x", label="Observed C in")
# ax[0].plot(df.index[df['is_obs_input_filled']], df["C in"][df['is_obs_input_filled']], ".", label="Filled C in")
ax[0].set_ylim(-0.5, 50)
ax[0].set_xlim(t.iloc[0], t[obs[-1]])

ax1_1 = ax[1].twinx()
ax1_1.plot(t, df["Q"], color="C0", alpha=0.3, label="Flow rate")

std = output_scenarios.iloc[burn_in:,valid_ind_out].std(axis=0)
upper = output_scenarios.iloc[burn_in:,valid_ind_out].mean(axis=0) + 1.96*std
lower = output_scenarios.iloc[burn_in:,valid_ind_out].mean(axis=0) - 1.96*std
ax[1].fill_between(t.values[valid_ind_out], lower, upper, color = "gray", alpha = 0.3)
ax[1].plot(t.values[valid_ind_out], output_scenarios.iloc[burn_in:,valid_ind_out].values.T,".", color = "gray", alpha = 0.01)
ax[1].plot(t.values[valid_ind_out],convolved[valid_ind_out], color = color_convol, markersize=1, label= "Convolution")
ax[1].plot(t.values[valid_ind_out], output_scenarios.iloc[burn_in:,valid_ind_out].mean(axis=0),"--" ,linewidth=1., color = "black", label = 'Prediction mean')
ax[1].plot(t, df["C out"], "x", color=color_data, label = 'Data', markersize=5.)

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
handles.extend([plt.Line2D([0], [0], color=color_data, lw =1.5)])
labels.extend(['Data'])


ax[0].legend(handles, labels, loc="upper right")

# Get existing legend handles and labels
handles, labels = ax[1].get_legend_handles_labels()
handles.extend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=5)])
labels.extend(['Prediction ensemble'])
handles.extend([plt.Line2D([0], [0], color='C0', lw =2, alpha = 0.5)])
labels.extend(['Flow rate'])

h = [handles[i] for i in [4,1,3,0,2]]
l = [labels[i] for i in [4,1,3,0,2]]

ax[1].legend(h, l, loc="upper right")
ax[1].set_xticks(t[obs[::10]])
for tick in ax[1].get_xticklabels():
    tick.set_rotation(45)

fig.tight_layout()
fig.savefig(f"{result_root}/input_output_{case_name}.pdf")

obs_ind = ~np.isnan(df["C out"].values[valid_ind_out])
observation = df["C out"].values[valid_ind_out]
observation = observation[obs_ind]

mse_conv = np.mean((convolved[valid_ind_out][obs_ind] - observation)**2)
mse_mean = np.mean((output_scenarios.iloc[burn_in:,valid_ind_out].mean(axis=0)[obs_ind] - observation)**2)

print(f"Mean squared error of convolution: {mse_conv}")
print(f"Mean squared error of mean: {mse_mean}")




# %%
# =============================================================================
# 2. Parameter posterior uncertainty
# =============================================================================

if case_name == "invariant_q_u_et_u":
    config = theta_invariant_q_u_et_u
elif case_name == "invariant_q_g_et_u":
    config = theta_invariant_q_g_et_u
elif case_name == "invariant_q_g_et_e":
    config = theta_invariant_q_g_et_e
elif case_name == "storage_q_g_et_u":
    config = theta_storage_q_g_et_u
elif case_name == "storage_q_g_et_u_changing_varying":
    config = theta_storage_q_g_et_u_changing

obs_uncertainty = config["obs_uncertainty"]

sigma_filled_paper = obs_uncertainty["sigma filled C in"]["prior_params"]
sigma_observed_paper = obs_uncertainty["sigma observed C in"]["prior_params"]
sigma_output_paper = obs_uncertainty["sigma C out"]["prior_params"]

a_paper = config["sas_specs"]["Q"]["Q SAS function"]["args"]["a"]
lambda_paper = config["scale_parameters"]["lambda"]["prior_params"]
sc_paper = config["scale_parameters"]["S_c"]["prior_params"]
etscale_paper = config["sas_specs"]["ET"]["ET SAS function"]["args"]["scale"]
c_old_paper = solute_parameters["C in"]["C_old"]

# %%
fig = plt.figure(figsize=(20, 8))
from matplotlib import gridspec
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1])

# First row
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])

# Second row
ax5 = fig.add_subplot(gs[1, 0])
ax6 = fig.add_subplot(gs[1, 1])
ax7 = fig.add_subplot(gs[1, 2])
ax8 = fig.add_subplot(gs[1, 3])

# Plotting
ax1.hist(theta[theta.columns[1]].values[burn_in:], 10, density=True, label="Posterior")
x = np.linspace(lambda_paper[0]-3*lambda_paper[1], lambda_paper[0]+3*lambda_paper[1], 1000)
loc, scale = lambda_paper[0], lambda_paper[1]
pdf = norm(loc, scale).pdf(x)
pdf[pdf<0] = 0.
ax1.plot(x, pdf, label="Prior", color='black')
mean = theta[theta.columns[1]].values[burn_in:].mean()
ax1.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Posterior mean')
ax1.set_ylabel("Density", fontsize=14)

ax2.hist(theta[theta.columns[2]].values[burn_in:], 10, density=True, label="Posterior")
x = np.linspace(sc_paper[0]-3*sc_paper[1], sc_paper[0]+3*sc_paper[1], 1000)
loc, scale = sc_paper[0], sc_paper[1]
pdf = norm(loc, scale).pdf(x)
pdf[pdf<0] = 0.
ax2.plot(x, pdf, label="Prior", color='black')
mean = theta[theta.columns[2]].values[burn_in:].mean()
ax2.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Posterior mean')

ax3.hist(theta[theta.columns[3]].values[burn_in:], 10, density=True, label="Posterior")
x = np.linspace(0.7*a_paper, 1.3*a_paper, 1000)
pdf = norm(a_paper, a_paper/10.).pdf(x)
ax3.plot(x, pdf, label="Prior", color='black')
mean = theta[theta.columns[3]].values[burn_in:].mean()
ax3.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Posterior mean')


ax4.hist(theta[theta.columns[4]].values[burn_in:], 10, density=True, label="Posterior")
x = np.linspace(0.7*etscale_paper, 1.3*etscale_paper, 1000)
pdf = norm(etscale_paper, etscale_paper/10.).pdf(x)
ax4.plot(x, pdf, label="Prior", color='black')
mean = theta[theta.columns[4]].values[burn_in:].mean()
ax4.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Posterior mean')

ax5.hist(theta[theta.columns[5]].values[burn_in:], 10, density=True, label="Posterior")
x = np.linspace(7./10.*c_old_paper, 13./10.*c_old_paper, 1000)
pdf = norm(c_old_paper, c_old_paper/10.).pdf(x)
ax5.plot(x, pdf, label="Prior", color='black')
mean = theta[theta.columns[5]].values[burn_in:].mean()
ax5.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Posterior mean')

ax6.hist(theta[theta.columns[6]].values[burn_in:], 10, density=True, label="Posterior")
x = np.linspace(sigma_observed_paper[0] - 3*sigma_observed_paper[1], sigma_observed_paper[0] + 3*sigma_observed_paper[1], 1000)
loc, scale = sigma_observed_paper[0], sigma_observed_paper[1]
pdf = norm(loc, scale).pdf(x)
pdf[pdf<0] = 0.
ax6.plot(x, pdf, label="Prior", color='black')
mean = theta[theta.columns[6]].values[burn_in:].mean()
ax6.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Posterior mean')

ax7.hist(theta[theta.columns[7]].values[burn_in:], 10, density=True, label="Posterior")
x = np.linspace(sigma_filled_paper[0] - 3*sigma_filled_paper[1], sigma_filled_paper[0] + 3*sigma_filled_paper[1], 1000)
loc, scale = sigma_filled_paper[0], sigma_filled_paper[1]
pdf = norm(loc, scale).pdf(x)
pdf[pdf<0] = 0.
ax7.plot(x, pdf, label="Prior", color='black')
mean = theta[theta.columns[7]].values[burn_in:].mean()
ax7.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Posterior mean')

ax8.hist(theta[theta.columns[8]].values[burn_in:], 10, density=True, label="Posterior")
x = np.linspace(sigma_output_paper[0] - 3*sigma_output_paper[1], sigma_output_paper[0] + 3*sigma_output_paper[1], 1000)
loc, scale = sigma_output_paper[0], sigma_output_paper[1]
pdf = norm(loc, scale).pdf(x)
pdf[pdf<0] = 0.
ax8.plot(x, pdf, label="Prior", color='black')
mean = theta[theta.columns[8]].values[burn_in:].mean()
ax8.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Posterior mean')


ax1.set_title(r"$\mu_{Q}$ scale parameter $\lambda$ (parameter $\theta_{x}$)", fontsize=14)
ax2.set_title(r"$\mu_{Q}$ scale parameter $S_c$ (parameter $\theta_{x}$)", fontsize=14)
ax3.set_title(r"$\mu_{Q}$ shape parameter $\alpha$ (parameter $\theta_{x}$)", fontsize=14)
ax4.set_title(r"$\mu_{ET}$ scale parameter (parameter $\theta_{x}$)", fontsize=14)
ax5.set_title(r"Initial condition $\mathcal{C}_{o}$ (parameter $\theta_{C}$)", fontsize=14)
ax6.set_title(r"Measured $C_J$ uncertainty $\sigma_m$ (parameter $\theta_{r}$)", fontsize=14)
ax7.set_title(r"Interpolated $C_J$ uncertainty $\sigma_i$ (parameter $\theta_{r}$)", fontsize=14)
ax8.set_title(r"Outflow uncertainty $\sigma_y$ (parameter $\theta_{y}$)", fontsize=14)

ax1.legend(frameon=False, fontsize=12, loc="upper left")
ax5.set_ylabel("Density", fontsize=14)

plt.tight_layout()
plt.show()
fig.savefig(f"{result_root}/parameter_posterior_{case_name}.pdf")


# %%
# =============================================================================
# 3. Plot time-varying parameters
# =============================================================================
# make sas function plot
a = theta['Q#@#Q SAS function#@#a'].values
lambda_ = theta['Q#@#Q SAS function#@#lambda'].values
sc = theta['Q#@#Q SAS function#@#S_c'].values
delta_S = df['S_scale_factor'].values

def make_plot(ax, a, lambda_, sc, delta_S, percentile, color):
    # find the index for delta_S that closest to given percentile 
    delta_S_percentile = np.percentile(delta_S, percentile)

    index = np.argmin(abs(delta_S - delta_S_percentile))

    # plot gamma function
    x = np.linspace(0, len(df), 1000)
    for i in range(len(a)):
        scale = lambda_[i] * (delta_S[index] - sc[i])

        cdf = gamma.cdf(x, a[i], 0, scale)
        ax.plot(x, cdf, color, alpha=0.08)


fig, axes = plt.subplots(1,2, figsize=(8, 4),sharey=True)
ax = axes[0]
ax1 = axes[1]
make_plot(ax, a[burn_in:], lambda_, sc, delta_S, 10,"C0")
make_plot(ax, a[burn_in:], lambda_, sc, delta_S, 50, "C1")
make_plot(ax, a[burn_in:], lambda_, sc, delta_S, 90, "C2")

# customize legend
handles, labels = ax.get_legend_handles_labels()
handles.extend([plt.Line2D([0], [0], color='C0', lw =2)])
labels.extend(['10th percentile'])
handles.extend([plt.Line2D([0], [0], color='C1', lw =2)])
labels.extend(['50th percentile'])
handles.extend([plt.Line2D([0], [0], color='C2', lw =2)])
labels.extend(['90th percentile'])
ax.legend(handles, labels, loc="upper left", frameon=False)

ax.set_title(r"SAS function for $\mathtt{Q}$: $\Omega_{\mathtt{Q}}$")
ax.set_xlabel(r"$S_T$")

b = theta['ET#@#ET SAS function#@#scale'].values
from scipy.stats import uniform
x = np.linspace(0, len(df), 1000)
for i in range(len(b)):
    cdf = uniform.cdf(x, 0, b[i])
    ax1.plot(x, cdf, 'C0', alpha=0.08)
ax1.set_title(r"SAS function for $\mathtt{ET}$: $\Omega_{\mathtt{ET}}$")
ax1.set_xlabel(r"$S_T$")
ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(f"{result_root}/sas_function_{case_name}.pdf")



# %%
