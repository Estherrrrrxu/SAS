# %%
import os

current_path = os.getcwd()
if current_path[-22:] != "tests_linear_reservoir":
    os.chdir("tests_linear_reservoir")
    print("Current working directory changed to 'tests_linear_reservoir'.")
import sys

sys.path.append("../")

from model.model_interface import ModelInterface
from model.ssm_model import SSModel
from model.your_model import LinearReservoir

from model.utils_chain import Chain
from functions.utils import plot_MLE
from functions.get_dataset import get_different_input_scenarios

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
num_input_scenarios = 30
num_parameter_samples = 30
len_parameter_MCMC = 10
plot_preliminary = False
start_ind = 0
unified_color = True
perfects = []

stn_input = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
for stn_i in stn_input:
    stn_o = stn_i * 3
    df = pd.read_csv(f"../Data/WhiteNoise/stn_{stn_i}_{stn_o}.csv", index_col=0)

    interval = [0, 20]

    (
        perfect,
        instant_gaps_2_d,
        instant_gaps_5_d,
        weekly_bulk,
        biweekly_bulk,
        weekly_bulk_true_q,
    ) = get_different_input_scenarios(df, interval, plot=False)

    case = perfect
    # Get data
    df = case.df
    df_obs = case.df_obs
    obs_made = case.obs_made
    case_name = case.case_name

    config = {"observed_made_each_step": obs_made, "outflux": "Q_true"}

    k_prior = [1.0, 0.3]
    initial_state_prior = [df_obs["Q_obs"][0], 0.01]
    sig_ipt_hat = df_obs["J_obs"].std(ddof=1) / stn_i
    input_uncertainty_prior = [sig_ipt_hat, sig_ipt_hat / 3.0]
    sig_obs_hat = df_obs["Q_obs"].std(ddof=1)
    obs_uncertainty_prior = [sig_obs_hat, sig_obs_hat / 3.0]

    # Save prior parameters
    path_str = f"../Results/TestLR/WhiteNoise/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}"
    if not os.path.exists(path_str):
        os.makedirs(path_str)

    prior_record = pd.DataFrame([k_prior, initial_state_prior, input_uncertainty_prior, obs_uncertainty_prior])
    prior_record.columns = ["mean", "std"]
    prior_record.index = ["k", "initial_state", "input_uncertainty", "obs_uncertainty"]
    prior_record.to_csv(f"{path_str}/prior_{stn_i}.csv")
    # %%
    theta_init = {
        "to_estimate": {
            "k": {
                "prior_dis": "normal",
                "prior_params": k_prior,
                "is_nonnegative": True,
            },
            "initial_state": {
                "prior_dis": "normal",
                "prior_params": initial_state_prior,
                "is_nonnegative": True,
            },
            "input_uncertainty": {
                "prior_dis": "normal",
                "prior_params": input_uncertainty_prior,
                "is_nonnegative": True,
            },
            "obs_uncertainty": {
                "prior_dis": "normal",
                "prior_params": obs_uncertainty_prior,
                "is_nonnegative": True,
            },
        },
        "not_to_estimate": {},
    }

    # %%
    # RUN MODEL
    # initialize model interface settings
    model_interface = ModelInterface(
        df=df_obs,
        customized_model=LinearReservoir,
        theta_init=theta_init,
        num_input_scenarios=num_input_scenarios,
        config=config,
    )

    if plot_preliminary:
        chain = Chain(model_interface=model_interface)
        chain.run_sequential_monte_carlo()
        plot_MLE(chain.state, df, df_obs, chain.pre_ind, chain.post_ind)
        plt.plot(chain.state.X.T, ".")

        chain.run_particle_MCMC_AS()
        plot_MLE(chain.state, df, df_obs, chain.pre_ind, chain.post_ind)
        plt.plot(chain.state.X.T, ".")

    # run PMCMC
    model = SSModel(
        model_interface=model_interface,
        num_parameter_samples=num_parameter_samples,
        len_parameter_MCMC=len_parameter_MCMC,
    )
    model.run_particle_Gibbs_SAEM()

    # %%
    # POST PROCESSING
    # get estimated parameters
    k = model.theta_record[:, 0]
    initial_state = model.theta_record[:, 1]
    input_uncertainty = model.theta_record[:, 2]
    obs_uncertainty = model.theta_record[:, 3]
    input_scenarios = model.input_record
    output_scenarios = model.output_record
    df_ipt = model_interface.df



    np.savetxt(f"{path_str}/k.csv",k)
    np.savetxt(f"{path_str}/initial_state.csv",initial_state)
    np.savetxt(f"{path_str}/input_uncertainty.csv",input_uncertainty)
    np.savetxt(f"{path_str}/obs_uncertainty.csv",obs_uncertainty)
    np.savetxt(f"{path_str}/input_scenarios.csv",input_scenarios)
    np.savetxt(f"{path_str}/output_scenarios.csv",output_scenarios)
    df_ipt.to_csv(f"{path_str}/df_ipt.csv")
# %%
