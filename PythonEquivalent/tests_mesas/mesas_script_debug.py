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
import seaborn as sns

# %%
# SET FOLDERS
# ================================================================
data_root = "/Users/esthersida/pMESAS/mesas"
result_root = "/Users/esthersida/pMESAS/mesas/Results"
if not os.path.exists(result_root):
    os.makedirs(result_root)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "-c", "--case-name", default="invariant_q_u_et_u", help="identify theta case"
)
args, unknown_args = parser.parse_known_args()
case_name = args.case_name


case_name = "storage_q_g_et_u"

start_ind, end_ind = 2250, 3383

num_input_scenarios = 5
num_parameter_samples = 5
len_parameter_MCMC = 1


df = pd.read_csv(f"{data_root}/data_preprocessed.csv", index_col=1, parse_dates=True)
data = df.iloc[start_ind:end_ind]
time = data.datetime
# data = data.reset_index(drop=True)


# %%
# RUN MODEL ================================================================
# initialize model settings
output_obs = data["C out"].notna().to_list()
config = {
    "observed_made_each_step": output_obs,
    "influx": ["J"],
    "outflux": ["C out"],
    "use_MAP_AS_weight": True,
    "use_MAP_ref_traj": True,
    "use_MAP_MCMC": True,
    "update_theta_dist": False,
}

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

plt.figure(figsize=(12, 4))
r = model_interface.R_prime
for i in range(num_input_scenarios):
    plt.scatter(
        time.iloc[r[i] > 0], r[i][r[i] > 0], marker=".", s=10, c="gray", alpha=0.5, label=f"Simulated inputs"
    )
obs = model_interface.df[model_interface.in_sol].to_numpy()
plt.plot(time.iloc[obs > 0], obs[obs > 0], ".", markersize=5, label="Observed inputs")
plt.yscale("log")
# plt.xticks(time[1::90], rotation=30) 
ax = plt.gca()
ax.set_xticks(ax.get_xticks()[1::90])
plt.xlim(time.iloc[0], time.iloc[-1])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[-2:], labels[-2:], loc="upper right", fontsize=12, ncol=2)
plt.ylabel("log(Concentration)")
plt.tight_layout()
plt.savefig(f"{result_root}/input_scenarios_{case_name}.pdf")

# r = r[:,:,0].T


# %%

chain = Chain(model_interface=model_interface)
chain.run_particle_filter_SIR()
# %%

plt.figure()
plt.plot(model_interface.df["C in"], "*")
plt.plot(model_interface.df.index, chain.state.R[:, :, 0].T, ".", markersize=0.7)

# %%
# for i in range(25):
#     plt.figure()
#     plt.plot(model_interface.df['C out'], "*")
#     plt.plot(model_interface.df.index, chain.state.Y[i,:,0].T)#, ".", markersize=0.7)
plt.figure()
plt.plot(model_interface.df["C out"], "*")
plt.plot(model_interface.df.index, chain.state.Y[:, :, 0].T, ".", markersize=0.7)
plt.plot(
    model_interface.df.index,
    chain.state.Y[np.argmax(chain.state.W), :, 0].T,
    ".",
    markersize=10,
)
print("done SIR check")
# %%

chain.run_particle_filter_AS()

plt.figure()
sns.boxplot(chain.state.R[:, :, 0])
plt.plot(model_interface.df["C in"], "*")
r = chain.state.R[:, :, 0].T
q = chain.state.Y[:, :, 0].T
plt.figure()
plt.plot(model_interface.df["C out"], "*")
plt.plot(model_interface.df.index, chain.state.Y[:, :, 0].T, ".", markersize=0.7)

print("done AS check")


# %%
# run actual particle Gibbs
model = SSModel(
    model_interface=model_interface,
    num_parameter_samples=num_parameter_samples,
    len_parameter_MCMC=len_parameter_MCMC,
)

model.run_particle_Gibbs()


# SAVE RESULTS ================================================================
# get estimated parameters
#
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
# # obs_ind = np.where(df['is_obs'])[0]
# plt.figure()
# # plt.plot(model_interface.df["J_true"], "k", linewidth=10)
# # plt.plot(model_interface.df["J"], "grey", linewidth=1)
# plt.plot(model_interface.df.index, r, "o")#,color="red")
# plt.plot(model_interface.df.index,input_scenarios[:, :].T, ".", color = 'grey')
# plt.plot(model_interface.df["C in"], "k_", linewidth=10)

#%%

plt.figure()
st, et = model_interface.observed_ind[0], model_interface.observed_ind[-1]
# plt.plot(model_interface.df["Q"], "grey")

time = model_interface.df.index
# plt.plot(time[st:et], output_scenarios[:, st:et].T,'.', alpha = 0.1, color="grey",lw=0.5, label="predicted")
# make a step plot
# plt.step(time[st:et], model_interface.df["C out"].backfill().iloc[st:et], label= "observed")
# plt.plot(time[st:et],model_interface.df["C out"].iloc[st:et], "*", label= "observed")
for i in range(len_parameter_MCMC + 1):
    plt.figure(figsize=(12, 4))

    plt.step(
        time[st:et],
        model_interface.df["C out"].backfill().iloc[st:et],
        label="observed",
    )
    plt.plot(
        time[st:et], output_scenarios[i, st:et].T, color="orange", label="predicted"
    )
    # plt.xlim([time[0], time[-1]])
#     plt.plot(model_interface.df.index[st:et], output_scenarios[i, st:et].T,   label=f"output {i}", lw=0.5)
# plt.legend(frameon=False)
# np.save(f"output.npy", output_scenarios)

# %%
# # save data as csv files
np.savetxt(f"{result_root}/qscale_{case_name}.csv", qscale, delimiter=",")
np.savetxt(f"{result_root}/etscale_{case_name}.csv", etscale, delimiter=",")
np.savetxt(f"{result_root}/c_old_{case_name}.csv", c_old, delimiter=",")
np.savetxt(f"{result_root}/sigma_observed_{case_name}.csv", sigma_observed, delimiter=",")
np.savetxt(f"{result_root}/sigma_filled_{case_name}.csv", sigma_filled, delimiter=",")
np.savetxt(f"{result_root}/sigma_output_{case_name}.csv", sigma_output, delimiter=",")
np.savetxt(f"{result_root}/input_scenarios_{case_name}.csv", input_scenarios, delimiter=",")
np.savetxt(f"{result_root}/output_scenarios_{case_name}.csv", output_scenarios, delimiter=",")


# %%
