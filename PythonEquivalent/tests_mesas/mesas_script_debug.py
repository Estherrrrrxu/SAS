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

from model.ssm_model import SSModel
from mesas.sas.model import Model as SAS_Model
import argparse

from mesas_cases import *

# %%
# SET FOLDERS
# ================================================================
data_root = "/Users/esthersida/pMESAS/mesas"
result_root = "/Users/esthersida/pMESAS/mesas/Results"
if not os.path.exists(result_root):
    os.makedirs(result_root)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-c","--case-name", default="invariant_q_u_et_u", help="identify theta case")
args, unknown_args = parser.parse_known_args()
case_name = args.case_name  # "invariant_q_u_et_u"

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

df["C in raw"][df["C in"] != df["C in raw"]] = ill_vals["C in"]
df["is_obs_input"][df["C in"] != df["C in raw"]] = ill_vals["is_obs"]
df["is_obs_input_filled"][df["C in"] != df["C in raw"]] = True

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
start_ind, end_ind = 0, 600
data = df.iloc[start_ind:end_ind]
# reset df index
data = data.reset_index(drop=True)
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

# RUN MODEL ================================================================
# initialize model settings
output_obs = data["C out"].notna().to_list()
config = {
    "observed_made_each_step": output_obs,
    "influx": ["J"],
    # "outflux": ["Q", "ET"],
    "outflux": ["C out"],
    "use_MAP_AS_weight": False,
    "use_MAP_ref_traj": False,
    "use_MAP_MCMC": False,
    "update_theta_dist": False,
}


num_input_scenarios = 5
num_parameter_samples = 5
len_parameter_MCMC = 5

model_interface_class = ModelInterfaceMesas

if case_name == "invariant_q_u_et_u":
    model_interface = model_interface_class(
        df=data,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_invariant_q_u_et_u,
    )
elif case_name == "invariant_q_g_et_u":
    model_interface = model_interface_class(
        df=data,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_invariant_q_g_et_u,
    )
elif case_name == "invariant_q_g_et_e":
    model_interface = model_interface_class(
        df=data,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_invariant_q_g_et_e,
    )
elif case_name == "storage_q_g_et_u":
    model_interface = model_interface_class(
        df=data,
        customized_model=SAS_Model,
        num_input_scenarios=num_input_scenarios,
        config=config,
        theta_init=theta_storage_q_g_et_u,
    )
else:
    raise ValueError("Case name not found.")
# %%

# check input scenarios generation
model_interface._bulk_input_preprocess()
# plt.figure()
# r = model_interface.R_prime
# for i in range(5):
#     plt.scatter(np.arange(r.shape[1]), r[i], marker=".", s=10)
# obs = model_interface.df[model_interface.in_sol].to_numpy()

# plt.plot(obs, "_")


# %%

chain = Chain(model_interface=model_interface)
chain.run_particle_filter_SIR()
#%%

plt.figure()
plt.plot(model_interface.df.index, chain.state.R[:,:,0].T, ".", markersize=0.5)
plt.plot(model_interface.df['C in'], "*")


plt.figure()
plt.plot(model_interface.df.index, chain.state.Y[:,:,0].T, ".", markersize=0.7)
plt.plot(model_interface.df['C out'], "*")

print('done SIR check')
# %%

chain.run_particle_filter_AS()

plt.figure()
plt.plot(model_interface.df['C out'], "*")


# fig, ax = plot_MAP(chain.state, df_obs, chain.pre_ind, chain.post_ind)
# ax[1].plot(chain.state.X.T, ".")





# print('done AS check')
#%%
# run actual particle Gibbs
model = SSModel(
    model_interface=model_interface,
    num_parameter_samples=num_parameter_samples,
    len_parameter_MCMC=len_parameter_MCMC,
)

model.run_particle_Gibbs()


# SAVE RESULTS ================================================================
# get estimated parameters
qscale = model.theta_record[:, 0]
etscale = model.theta_record[:, 1]
c_old = model.theta_record[:, 2]
sigma_observed = model.theta_record[:, 3]
sigma_filled = model.theta_record[:, 4]
sigma_output = model.theta_record[:, 5]

input_scenarios = model.input_record
output_scenarios = model.output_record
df = model_interface.df


# %%
# obs_ind = np.where(df['is_obs'])[0]
plt.figure()
# plt.plot(model_interface.df["J_true"], "k", linewidth=10)
# plt.plot(model_interface.df["J"], "grey", linewidth=1)
plt.plot(model_interface.df.index,input_scenarios[:, :].T, '.')
plt.plot(model_interface.df["C in"], "k_", linewidth=10)

plt.figure()
# plt.plot(model_interface.df["Q"], "grey")
plt.plot(model_interface.df["C out"], "*", label= "observed")
for i in range(6):
    plt.scatter(model_interface.df.index, output_scenarios[i, :].T, marker='.', label=f"output {i}", s=1)
plt.legend(frameon=False)

# %%
# # save data as csv files
# np.savetxt(f"{result_root}/qscale_{case_name}.csv", qscale, delimiter=",")
# np.savetxt(f"{result_root}/etscale_{case_name}.csv", etscale, delimiter=",")
# np.savetxt(f"{result_root}/c_old_{case_name}.csv", c_old, delimiter=",")
# np.savetxt(f"{result_root}/sigma_observed_{case_name}.csv", sigma_observed, delimiter=",")
# np.savetxt(f"{result_root}/sigma_filled_{case_name}.csv", sigma_filled, delimiter=",")
# np.savetxt(f"{result_root}/sigma_output_{case_name}.csv", sigma_output, delimiter=",")
# np.savetxt(f"{result_root}/input_scenarios_{case_name}.csv", input_scenarios, delimiter=",")
# np.savetxt(f"{result_root}/output_scenarios_{case_name}.csv", output_scenarios, delimiter=",")




# %%
