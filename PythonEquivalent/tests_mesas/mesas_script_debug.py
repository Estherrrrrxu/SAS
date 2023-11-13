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

# %%
# settings that are not likely to change
ipt_mean = 5.0
test_case = "WhiteNoise"
data_root = "/Users/esthersida/pMESAS"

stn_input = [3]

length = 3000

model_interface_class = ModelInterfaceMesas
obs_pattern = "bulk output" 

# %%
for stn_i in stn_input:
    # Read data and chop data
    df = pd.read_csv(
        f"{data_root}/Data/{test_case}/stn_{stn_i}_T_{length}_k_{k}_mean_{ipt_mean}_std_{ipt_std}.csv",
        index_col=0,
    )
    df = df.iloc[interval[0] : interval[1], :]

    # get specific running case
    case = get_different_input_scenarios(
        df, interval, plot=False, observation_mode=obs_mode
    )

    df_obs = case.df_obs
    obs_made = case.obs_made
    case_name = case.case_name

    # Create result path
    path_str = f"{data_root}/Results/TestLR/{test_case}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}_k_{k}_mean_{ipt_mean}_std_{ipt_std}_length_{interval[1]-interval[0]}/{case_name}"
    if not os.path.exists(path_str):
        os.makedirs(path_str)

    # Set prior parameters
    k_prior = [k, k / 3.0]
    initial_state_prior = [df_obs["Q_obs"][0], df_obs["Q_obs"][0] / 3. * np.sqrt(k)]
    sig_ipt_hat = df_obs["J_obs"].std(ddof=1) 
    input_uncertainty_prior = [sig_ipt_hat, sig_ipt_hat/ 3.0]
    sig_obs_hat = df_obs["Q_obs"].std(ddof=1) / stn_i
    obs_uncertainty_prior = [sig_obs_hat, sig_obs_hat/ 3.0]
    # Observation is set to be very small because currently using Q_true as the observation 

    influx_type = "J_true"
    outflux_type = "Q_true"
    
    config = {
        "observed_made_each_step": obs_made,
        "influx": influx_type+'_fine',
        "outflux": outflux_type,
        "use_MAP_AS_weight": False,
        "use_MAP_ref_traj": False,
        "use_MAP_MCMC": False,
        "update_theta_dist": False,
    }


    # Save prior parameters and compile
    prior_record = pd.DataFrame(
        [k_prior, initial_state_prior, input_uncertainty_prior, obs_uncertainty_prior]
    )

    prior_record.columns = ["mean", "std"]
    prior_record.index = ["k", "initial_state", "input_uncertainty", "obs_uncertainty"]
    prior_record.to_csv(f"{path_str}/prior_parameters_{stn_i}.csv")

    run_parameter = [num_input_scenarios, num_parameter_samples, len_parameter_MCMC]
    plot_preliminary = True
    #%%
    # SET PRIOR PARAMETERS ================================================================
    # Unpack prior parameters
    k_prior = prior_record.loc["k"].values.tolist()
    initial_state_prior = prior_record.loc["initial_state"].values.tolist()
    input_uncertainty_prior = prior_record.loc["input_uncertainty"].values.tolist()
    obs_uncertainty_prior = prior_record.loc["obs_uncertainty"].values.tolist()
    # Set theta_init
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

    # RUN MODEL ================================================================
    # initialize model settings
    num_input_scenarios = run_parameter[0]
    num_parameter_samples = run_parameter[1]
    len_parameter_MCMC = run_parameter[2]

    model_interface = model_interface_class(
        df=df_obs,
        customized_model=LinearReservoir,
        theta_init=theta_init,
        num_input_scenarios=num_input_scenarios,
        config=config,
    )
    # test model run with given settings
    if plot_preliminary:
        chain = Chain(model_interface=model_interface)
        chain.run_particle_filter_SIR()
        fig, ax = plot_MAP(chain.state, df_obs, chain.pre_ind, chain.post_ind)
        ax[1].plot(chain.state.X.T, ".")

        chain.run_particle_filter_AS()
        fig, ax = plot_MAP(chain.state, df_obs, chain.pre_ind, chain.post_ind)
        ax[1].plot(chain.state.X.T, ".")
    #%%
    # run actual particle Gibbs
    model = SSModel(
        model_interface=model_interface,
        num_parameter_samples=num_parameter_samples,
        len_parameter_MCMC=len_parameter_MCMC,
    )
    st = time.time()
    model.run_particle_Gibbs()
    model_run_time = time.time() - st

    # SAVE RESULTS ================================================================
    # get estimated parameters
    k = model.theta_record[:, 0]
    initial_state = model.theta_record[:, 1]
    input_uncertainty = model.theta_record[:, 2]
    obs_uncertainty = model.theta_record[:, 3]
    input_scenarios = model.input_record

    output_scenarios = model.output_record
    df = model_interface.df



# %%
obs_ind = np.where(df['is_obs'])[0]
plt.plot(model_interface.df["J_true"], "k", linewidth=10)
plt.plot(input_scenarios[:, :].T, marker='.')
# %%
plt.plot(model_interface.df["Q_true"], "k", linewidth=10)
plt.plot(output_scenarios[:, :].T, marker='.')
plt.ylim([4, 5.5])
# %%
