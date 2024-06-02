# %%
import os

current_path = os.getcwd()

if current_path[-11:] != "tests_mesas":
    os.chdir("tests_mesas")
    print("Current working directory changed to 'tests_mesas'.")
import sys

sys.path.append("../")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from mesas.sas.model import Model as SAS_Model
from mesas_cases import theta_invariant_q_u_et_u, theta_storage_q_g_et_u

# %%
# SET FOLDERS
# ================================================================
data_root = "/Users/esthersida/pMESAS/mesas"
result_root = "/Users/esthersida/pMESAS/mesas/Results/Results"

if not os.path.exists(result_root):
    os.makedirs(result_root)

df = pd.read_csv(f"{data_root}/data_preprocessed.csv", index_col=1, parse_dates=True)
data = df.iloc[2250:3383]
case_names = ["invariant_q_u_et_u",  "storage_q_g_et_u"]

for case_name in case_names:
    # set up config
    config = {}
    if case_name == "invariant_q_u_et_u":
        config['solute_parameters'] = theta_invariant_q_u_et_u['solute_parameters']
        config['sas_specs'] = theta_invariant_q_u_et_u['sas_specs']
        config['options'] = theta_invariant_q_u_et_u['options']
    elif case_name == "storage_q_g_et_u":
        config['solute_parameters'] = theta_storage_q_g_et_u['solute_parameters']
        config['sas_specs'] = theta_storage_q_g_et_u['sas_specs']
        config['options'] = theta_storage_q_g_et_u['options']
    else:
        raise ValueError("Case name not recognized.")

    # Create the model for getting C_Q
    model = SAS_Model(data,
                config=deepcopy(config),
                verbose=False
                )

    sas_specs = config["sas_specs"]
    solute_parameters = config["solute_parameters"]
    options = config["options"]

    # Run the model
    model.run()

    # Extract results
    data_df = model.data_df.copy()

    J = data_df['J'].to_numpy()
    Q = data_df['Q'].to_numpy()

    timeseries_length = len(J)
    dt = model.options['dt']
    C_old = model.solute_parameters['C in']['C_old']

    # 
    # this is the part creating fake data for calculating evpoconc_factor
    data_unit_C_J = data.copy()
    a = 1.
    data_unit_C_J['C in'] = a
    config_copy = deepcopy(config)
    config_copy['solute_parameters']['C in']['C_old'] = a
    model_conv = SAS_Model(data_unit_C_J,
                    config=deepcopy(config_copy),
                    verbose=False,
                    )
    model_conv.run()

    # 
    # correcting CT timestep
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

    # 
    # convolution the same as what I proposed in integration

    C_J = data_df['C in'].to_numpy()
    C_Q = np.zeros(timeseries_length)
    C_old = 7.11

    # pQ back
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
            ti = t-T
            C_Q_test[t] += C_J[ti] * pQ[T, t] * evapoconc_factor[T, t] * dt
        C_Q_test[t] += C_old * (1 - pQ[:t + 1, t].sum() * dt)
        _start_ind += 1
        _end_ind += 1

    # 
    # plot for double check
    plt.figure()
    plt.plot(C_Q, label = 'C_Q from convolution')
    plt.plot(C_Q_test, ":" ,label = 'C_Q using another method')
    plt.plot(data_df['C in --> Q'].to_numpy(), ":", label = 'C_Q from model')
    plt.plot(data_df['C out'].to_numpy(), "*", label = 'Actual C_Q')
    plt.legend(frameon = False)
    np.savetxt(f"{result_root}/C_Q_conv_{case_name}.csv", C_Q, delimiter=",")
    


# %%
