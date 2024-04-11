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
plt.figure()
plt.plot(ill_vals["C in"], "_", label="Replaced ill values")
plt.plot(ill_vals["C in"][ill_vals["is_obs"]], "x", label="Assume is observed")
plt.xlim([ill_vals.index[20], ill_vals.index[100]])
plt.legend(frameon=False)
plt.ylabel("C in [mg/L]")
plt.title("Replaced bulk mean and observed timestamp check")


plt.figure()
plt.plot((df["C in raw"] * df["J"]).cumsum(), label="Simple backfill raw data")
# replace the ill values under new condition
df["C in raw"][df["C in"] != df["C in raw"]] = ill_vals["C in"]
df["is_obs_input"][df["C in"] != df["C in raw"]] = ill_vals["is_obs"]
df["is_obs_input_filled"][df["C in"] != df["C in raw"]] = True
plt.plot((df["C in raw"] * df["J"]).cumsum(), label="Recovered raw data")
plt.plot((df["C in"] * df["J"]).cumsum(), label="Published data")
plt.ylabel("Cumulative mass balance [mg]")
plt.legend(frameon=False)
plt.title("Cumulative mass balance check after replacement")

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
from mesas.sas.model import Model as SAS_Model
from mesas_cases import theta_invariant_q_u_et_u
from copy import deepcopy
# %%
config_invariant_q_u_et_u = {}
config_invariant_q_u_et_u['solute_parameters'] = theta_invariant_q_u_et_u['solute_parameters']
config_invariant_q_u_et_u['sas_specs'] = theta_invariant_q_u_et_u['sas_specs']
config_invariant_q_u_et_u['options'] = theta_invariant_q_u_et_u['options']
config_invariant_q_u_et_u['options']['record_state'] = True



# %%

fake_data = df.iloc[:500]
data = fake_data.copy()

# Create the model for getting C_Q
model = SAS_Model(data,
              config=deepcopy(config_invariant_q_u_et_u),
              verbose=False
              )

sas_specs = theta_invariant_q_u_et_u["sas_specs"]
solute_parameters = theta_invariant_q_u_et_u["solute_parameters"]
options = theta_invariant_q_u_et_u["options"]

# Run the model
model.run()

# Extract results
data_df = model.data_df.copy()

J = data_df['J'].to_numpy()
Q = data_df['Q'].to_numpy()

timeseries_length = len(J)
dt = model.options['dt']
C_old = model.solute_parameters['C in']['C_old']
# %%
# this is the part creating fake data
data_unit_C_J = data.copy()
a = 1.
data_unit_C_J['C in'] = a
config_invariant_q_u_et_u_conv = deepcopy(config_invariant_q_u_et_u)
config_invariant_q_u_et_u_conv['solute_parameters']['C in']['C_old'] = a
model_conv = SAS_Model(data_unit_C_J,
                config=deepcopy(config_invariant_q_u_et_u_conv),
                verbose=False,
                )
model_conv.run()


# %%

CT = model_conv.get_CT('C in')
CT[np.isnan(CT)] = 1.

# Create a new_CT array with the same shape and type as CT
new_CT = np.ones_like(CT)
# Update the first row of new_CT
new_CT[0] = (CT[0] + 1.) / 2.
# Update the remaining elements
for i in range(1, CT.shape[0]):
    for j in range(i, CT.shape[1]):
        new_CT[i, j] = (CT[i-1, j-1] + CT[i, j]) / 2.

evapoconc_factor = new_CT[:,1:]
# evapoconc_factor = CT[:,1:]
#%%
C_J = data_df['C in'].to_numpy()
#%%
C_Q = np.zeros(timeseries_length)
C_old = model.solute_parameters['C in']['C_old']
# pQback
pQ = model.get_pQ(flux='Q')


for t in range(timeseries_length):

    # the maximum age is t
    for T in range(t+1):
        # the entry time is ti
        ti = t-T
        C_Q[t] += C_J[ti]*pQ[T,t]*evapoconc_factor[T,t]*dt

    C_Q[t] += C_old * (1-pQ[:t+1,t].sum()*dt)

C_Q_test = np.zeros(timeseries_length)
_start_ind = 0
_end_ind = 1

while _end_ind < timeseries_length:
    # actual time is t
    t = _start_ind
    # the maximum age is t
    for T in range(_end_ind + 1):
        # the entry time is ti
        ti = t - T
        C_Q_test[t] += C_J[ti] * pQ[T, t] * evapoconc_factor[T, t] * dt
    C_Q_test[t] += C_old * (1 - pQ[:t + 1, t].sum() * dt)
    _start_ind += 1
    _end_ind += 1

#%%
plt.figure()
plt.plot(C_Q, label = 'C_Q from convolution')
plt.plot(data_df['C in --> Q'].to_numpy(), ":", label = 'C_Q from model')
plt.plot(C_Q_test, "--", label = 'C_Q test')
plt.plot(data_df['C out'].to_numpy(), "*", label = 'Actual C_Q')
plt.legend(frameon = False)
plt.xlim([0,500])
plt.figure()
plt.plot(data_df['C in --> Q'].to_numpy() - C_Q, label = 'C_Q from model - C_Q from convolution')
# %%

