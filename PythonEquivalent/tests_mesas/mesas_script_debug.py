# %%
import os

current_path = os.getcwd()

if current_path[-11:] != "tests_mesas":
    os.chdir("tests_mesas")
    print("Current working directory changed to 'tests_mesas'.")
import sys

sys.path.append("../")

from functions.get_dataset import get_different_input_scenarios
import pandas as pd

from tests_mesas.mesas_interface import ModelInterfaceMesas
import matplotlib.pyplot as plt
import numpy as np
from model.utils_chain import Chain
from functions.utils import plot_MAP

# %%
# SET FOLDERS
# ================================================================
data_root = "/Users/esthersida/pMESAS/mesas"
result_root = "/Users/esthersida/pMESAS/mesas/Results"
if not os.path.exists(result_root):
    os.makedirs(result_root)

# %%
# READ DATA AND PREPROCESS
# ================================================================

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

# %%
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

# sanity check
# plt.figure()
# plt.plot(ill_vals["C in"], "_", label="Replaced ill values")
# plt.plot(ill_vals["C in"][ill_vals["is_obs"]], "x", label="Assume is observed")
# plt.xlim([ill_vals.index[20], ill_vals.index[100]])
# plt.legend(frameon=False)
# plt.ylabel("C in [mg/L]")
# plt.title("Replaced bulk mean and observed timestamp check")


# plt.figure()
# plt.plot((df["C in raw"] * df["J"]).cumsum(), label="Simple backfill raw data")
# # replace the ill values under new condition
df["C in raw"][df["C in"] != df["C in raw"]] = ill_vals["C in"]
df["is_obs_input"][df["C in"] != df["C in raw"]] = ill_vals["is_obs"]
df["is_obs_input_filled"][df["C in"] != df["C in raw"]] = True
# plt.plot((df["C in raw"] * df["J"]).cumsum(), label="Recovered raw data")
# plt.plot((df["C in"] * df["J"]).cumsum(), label="Published data")
# plt.ylabel("Cumulative mass balance [mg]")
# plt.legend(frameon=False)
# plt.title("Cumulative mass balance check after replacement")


# %%
df["is_obs_ouput"] = df["C out"].notna()
df.columns = [
    "timestep",
    "J",
    "C in fake",
    "Q",
    "ET",
    "C out",
    "S_scale",
    "is_obs_input",
    "is_obs_input_filled",
    "C in",
    "is_obs_output",
]
# %%
data = df.iloc[:500]
# fig, ax = plt.subplots(4, 1, figsize=(12, 12))
# # precipitation
# J = ax[0].bar(data.index, data["J"], label="J")

# ax0p = ax[0].twinx()
# CJ = ax0p.scatter(
#     data.index[data["C in"] > 0],
#     data["C in"][data["C in"] > 0],
#     color="r",
#     marker="x",
#     label=r"$Observed\ bulk\ C_J$",
# )
# temp_ind = np.logical_and(data["is_obs_input_filled"].values, data["C in"].values > 0)
# CJ1 = ax0p.scatter(
#     data.index[temp_ind],
#     data["C in"][temp_ind],
#     color="green",
#     marker="x",
#     label=r"$Filled\ C_J$",
# )


# discharge
# Q = ax[1].plot(data.index, data["Q"], label="Q")

# ax1p = ax[1].twinx()
# CQ = ax1p.scatter(
#     data.index[data["C out"] > 0],
#     data["C out"][data["C out"] > 0],
#     color="r",
#     marker="x",
#     label=r"$C_Q$",
# )

# # evapotranspiration
# ax[2].plot(data["ET"])

# # storage
# ax[3].plot(data["S_scale"])

# # settings
# ax[0].set_title("Input - Precipitation", fontsize=16)
# ax[1].set_title("Output 1 - Discharge", fontsize=16)
# ax[2].set_title("Output 2 - Evapotranspiration (ET)", fontsize=16)
# ax[3].set_title("State - Maximum storage", fontsize=16)

# ax[0].set_ylabel("Precipitation [mm/d]", fontsize=14)
# ax0p.set_ylabel("Concentration [mg/L]", fontsize=14)
# ax[0].set_ylim([data["J"].max() * 1.02, data["J"].min()])
# ax0p.set_ylim([data["C in"].max() * 1.2, 0.0])

# ax[1].set_ylabel("Discharge [mm/d]", fontsize=14)
# ax1p.set_ylabel("Concentration [mg/L]", fontsize=14)

# ax[2].set_ylabel("ET [mm/h]", fontsize=14)
# ax[3].set_ylabel("S max [mm]", fontsize=14)


# lines = [J, CJ, CJ1]
# labels = [line.get_label() for line in lines]
# ax[0].legend(lines, labels, frameon=False, loc="lower left", fontsize=14)

# lines = [Q[0], CQ]
# labels = [line.get_label() for line in lines]
# ax[1].legend(lines, labels, frameon=False, loc="upper left", fontsize=14)


# fig.tight_layout()


# %%
from mesas.sas.model import Model as SAS_Model
from mesas_cases import *


# %%

# Create the model
# model = Model(data,
#               config=config_invariant_q_u_et_u,
#               verbose=False
#               )

# sas_specs = sas_specs_invariant_q_u_et_u
# solute_parameters = theta_invariant_q_u_et_u["soluhttps://file+.vscode-resource.vscode-cdn.net/Users/esthersida/opt/anaconda3/envs/linear_system/lib/python3.10/site-packages/mesas/sas/model.py:557te_parameters"]
# options = theta_invariant_q_u_et_u["options"]

# # Run the model
# model.run()

# # Extract results
# data_df = model.data_df
# flux = model.fluxorder[0]

# #%%
# data_unit_C_J = data.copy()
# a = 1.
# data_unit_C_J['C in'] = a
# config_invariant_q_u_et_u_conv = theta_invariant_q_u_et_u.copy()
# config_invariant_q_u_et_u_conv['solute_parameters']['C in']['C_old'] = a
# model_conv = Model(data_unit_C_J,
#                 config=config_invariant_q_u_et_u_conv,
#                 verbose=False,
#                 )
# model_conv.run()

# # %%
# # Check convolution
# # data
# C_J = data_df['C in'].to_numpy()
# timeseries_length = len(C_J)
# dt = model.options['dt']
# C_old = model.solute_parameters['C in']['C_old']
# Q = data_df['Q'].to_numpy()
# J = data_df['J'].to_numpy()

# #%%
# C_Q = np.zeros(timeseries_length)
# # pQback
# pQ = model.get_pQ(flux='Q')
# evapoconc_factor = model_conv.get_CT('C in')
# evapoconc_factor[np.isnan(evapoconc_factor)] = 1.

# for t in range(timeseries_length):

#     # the maximum age is t
#     for T in range(t+1):
#         # the entry time is ti
#         ti = t-T
#         C_Q[t] += C_J[ti]*pQ[T,t]*evapoconc_factor[T,t]*dt

#     C_Q[t] += C_old * (1-pQ[:t+1,t].sum()*dt)

# plt.figure()
# plt.plot(C_Q, label = 'C_Q from convolution')
# plt.plot(data_df['C in --> Q'].to_numpy(), ":", label = 'C_Q from model')


# plt.legend(frameon = False)


# %%

# RUN MODEL ================================================================
# initialize model settings
output_obs = data["C out"].notna().to_list()
config = {
    "observed_made_each_step": output_obs,
    "influx": ["J"],
    "outflux": ["Q", "ET"],
    "use_MAP_AS_weight": False,
    "use_MAP_ref_traj": False,
    "use_MAP_MCMC": False,
    "update_theta_dist": False,
}

num_input_scenarios = 5
num_parameter_samples = 5
len_parameter_MCMC = 5

model_interface_class = ModelInterfaceMesas

model_interface = model_interface_class(
    df=data,
    customized_model=SAS_Model,
    num_input_scenarios=num_input_scenarios,
    config=config,
    theta_init=theta_invariant_q_u_et_u,
)

# %%
# check input scenarios generation
# model_interface._bulk_input_preprocess()
# r = model_interface.R_prime
# for i in range(5):
#     plt.scatter(np.arange(r.shape[1]), r[i], marker=".", s=10)
# obs = model_interface.df[model_interface.in_sol].to_numpy()
# obs[model_interface.influx == 0.0] = 0.0
# plt.plot(obs, "_")

# %%
Rt = model_interface.input_model(0, 16)
self = model_interface
num_iter = Rt.shape[1]
Xt = np.zeros((self.N, num_iter, self.num_states))


# %%

chain = Chain(model_interface=model_interface)
chain.run_particle_filter_SIR()
# fig, ax = plot_MAP(chain.state, df_obs, chain.pre_ind, chain.post_ind)
# ax[1].plot(chain.state.X.T, ".")
# %%

#         chain.run_particle_filter_AS()
#         fig, ax = plot_MAP(chain.state, df_obs, chain.pre_ind, chain.post_ind)
#         ax[1].plot(chain.state.X.T, ".")
#     #%%
#     # run actual particle Gibbs
#     model = SSModel(
#         model_interface=model_interface,
#         num_parameter_samples=num_parameter_samples,
#         len_parameter_MCMC=len_parameter_MCMC,
#     )
#     st = time.time()
#     model.run_particle_Gibbs()
#     model_run_time = time.time() - st

#     # SAVE RESULTS ================================================================
#     # get estimated parameters
#     k = model.theta_record[:, 0]
#     initial_state = model.theta_record[:, 1]
#     input_uncertainty = model.theta_record[:, 2]
#     obs_uncertainty = model.theta_record[:, 3]
#     input_scenarios = model.input_record

#     output_scenarios = model.output_record
#     df = model_interface.df


# # %%
# obs_ind = np.where(df['is_obs'])[0]
# plt.plot(model_interface.df["J_true"], "k", linewidth=10)
# plt.plot(input_scenarios[:, :].T, marker='.')
# # %%
# plt.plot(model_interface.df["Q_true"], "k", linewidth=10)
# plt.plot(output_scenarios[:, :].T, marker='.')
# plt.ylim([4, 5.5])
# %%
