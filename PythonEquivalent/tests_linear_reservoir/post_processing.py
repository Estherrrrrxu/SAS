    



# %%
num_input_scenarios = 30
num_parameter_samples = 25
len_parameter_MCMC = 20
k = 1.
ipt_std = 0.1
obs_mode = "deci_2d"
interval = [0, 40]

k_true = 1.0
initial_state_true = 5.0
input_uncertainty_true = 0.1
obs_uncertainty_true = 0.0
dt = 1.0 

sig_e = dt / stn_i
phi = 1 - k_true * dt
sig_q = np.sqrt(sig_e**2 / (1 - phi**2))



# %%




path_str = f"../Results/TestLR/{test_case}/{stn_i}_N_{num_input_scenarios}_D_{num_parameter_samples}_L_{len_parameter_MCMC}/{case_name}"

k = np.loadtxt(f"{path_str}/k.csv")
initial_state = np.loadtxt(f"{path_str}/initial_state.csv")
input_uncertainty = np.loadtxt(f"{path_str}/input_uncertainty.csv")
obs_uncertainty = np.loadtxt(f"{path_str}/obs_uncertainty.csv")
input_scenarios = np.loadtxt(f"{path_str}/input_scenarios.csv")
output_scenarios = np.loadtxt(f"{path_str}/output_scenarios.csv")

threshold = 5
plot_df = pd.DataFrame(
    {
        "k": k,
        "initial state": initial_state,
        "input uncertainty": input_uncertainty,
        "obs uncertainty": obs_uncertainty,
    }
)



true_params = [
    k_true,
    initial_state_true,
    input_uncertainty_true,
    obs_uncertainty_true,
    sig_q,
]

prior_params = pd.read_csv(f"{path_str}/prior_parameters_{stn_i}.csv", index_col=0)

# plot posterior
g = plot_parameter_posterior(plot_df, true_params, prior_params, threshold)
g.savefig(f"{path_str}/posterior.pdf")

# plot trajectories
estimation = {"input": input_scenarios, "output": output_scenarios}
truth_df = pd.read_csv(f"{path_str}/df.csv", index_col=0)

g = plot_scenarios(truth_df, estimation, threshold, stn_i, sig_q)
g.savefig(f"{path_str}/scenarios.pdf")

g = plot_scenarios(truth_df, estimation, threshold, stn_i, sig_q, line_mode=True)
g.savefig(f"{path_str}/scenarios_line.pdf")

RMSE_J = np.sqrt(np.mean((estimation["input"][threshold:,1:].mean(axis=0) - truth_df["J_true"][1:])**2))
RMSE_Q = np.sqrt(np.mean((estimation["output"][threshold:,:].mean(axis=0) - truth_df["Q_true"][:])**2))
print(f"RSME_J: {RMSE_J}, theoretical: {input_uncertainty_true}")
print(f"RSME_Q: {RMSE_Q}, theoretical: {obs_uncertainty_true}")

# calculate KL divergence
true_q_mean = truth_df["Q_true"].to_numpy()
true_q_std = sig_q
obs_q_mean = output_scenarios[threshold:,].mean(axis=0)
obs_q_std = output_scenarios[threshold:,].std(axis=0)

KL = cal_KL(true_q_std, obs_q_std, true_q_mean, obs_q_mean)
# plot KL divergence
plt.figure()
plt.plot(KL[:-1])
plt.xlabel("Timestep")
plt.ylabel("KL divergence")
plt.show()
plt.savefig(f"{path_str}/KL_divergence.pdf")

# convergence check
convergence_check_plot(plot_df, 100)

# %%
