# %%
from model.model_interface import ModelInterface
from model.ssm_model import SSModel
from model.your_model import LinearReservoir
from model.utils_chain import Chain

from functions.utils import plot_MAP
import numpy as np
import pandas as pd
from typing import Optional, List

from numba import vectorize
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

# %%
def run_with_given_settings(df_obs: pd.DataFrame, config: dict, run_parameter: List[int], path_str: str, prior_record: pd.DataFrame, plot_preliminary: Optional[bool] = True, model_interface_class: Optional[ModelInterface]=ModelInterface) -> None:
    """model_run_routine

    Args:
        path_str (str): the path to save results
        prior_record (pd.DataFrame): the prior parameters to sample from
        plot_preliminary (bool, optional): whether to test run algorithms. Defaults to True.
    """
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
    #
    # run actual particle Gibbs
    model = SSModel(
        model_interface=model_interface,
        num_parameter_samples=num_parameter_samples,
        len_parameter_MCMC=len_parameter_MCMC,
    )
    model.run_particle_Gibbs()

    # SAVE RESULTS ================================================================
    # get estimated parameters
    k = model.theta_record[:, 0]
    initial_state = model.theta_record[:, 1]
    input_uncertainty = model.theta_record[:, 2]
    obs_uncertainty = model.theta_record[:, 3]
    input_scenarios = model.input_record
    output_scenarios = model.output_record
    df = model_interface.df

    np.savetxt(f"{path_str}/k.csv",k)
    np.savetxt(f"{path_str}/initial_state.csv",initial_state)
    np.savetxt(f"{path_str}/input_uncertainty.csv",input_uncertainty)
    np.savetxt(f"{path_str}/obs_uncertainty.csv",obs_uncertainty)
    np.savetxt(f"{path_str}/input_scenarios.csv",input_scenarios)
    np.savetxt(f"{path_str}/output_scenarios.csv",output_scenarios)
    df.to_csv(f"{path_str}/df.csv")

    print(f"Results saved to {path_str}.")
    return None

# %%
# %%
@vectorize(["float64(float64, float64, float64, float64)"])
def cal_KL(sigma1, sigma2, mu1, mu2):
    return (
        np.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
        - 1 / 2
    )


# %%
def normal_pdf(
    mean: float,
    std: float,
    size: Optional[int] = 1000,
    nonnegative: Optional[bool] = True,
) -> List[np.ndarray]:
    """Generate normal pdf for given parameters in the range of 3 standard deviations.

    Args:
        mean (float): mean
        std (float): standard deviation
        size (Optional[int]): num of segments for generating pdf. Defaults to 1000.
        nonnegative (Optional[bool]): if the parameter is non-neg. Defaults to True.

    Returns:
        List[np.ndarray, np.ndarray]: x-axis and pdf
    """

    # Generate a range of x values
    x = np.linspace(mean - 3 * std, mean + 3 * std, size)

    # Calculate the probability density function (PDF) for the given parameters
    if nonnegative:
        pdf = ss.truncnorm.pdf(
            x, (0 - mean) / std, (np.inf - mean) / std, loc=mean, scale=std
        )
    else:
        pdf = ss.norm.pdf(x, loc=mean, scale=std)
    return x, pdf


# %%
# Posterior plots
def plot_parameter_posterior(
    plot_df: pd.DataFrame,
    true_params: List[float],
    prior_params: pd.DataFrame,
    threshold: int,
    levels: Optional[int] = 10,
    label_fontsize: Optional[int] = 14,
) -> plt.Figure:
    """Plots paired plot of the posterior distribution of parameters.

    Args:
        plot_df (pd.DataFrame): Parameter trajectories
        true_params (List[float]): The true parameters or approximated true parameters
        threshold (int): Cutoff threshold for stablized MCMC samples
    """
    # chop according to threshold
    plot_df = plot_df.iloc[threshold:]

    # read true parameters
    k_true = true_params[0]
    initial_state_true = true_params[1]
    input_uncertainty_true = true_params[2]
    obs_uncertainty_true = true_params[3]

    # make baseline plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    # remove upper triangle
    for i in range(4):
        for j in range(4):
            if i < j:
                axes[i, j].axis("off")

    # add prior distribution
    k_prior = prior_params.loc["k"].to_list()
    initial_state_prior = prior_params.loc["initial_state"].to_list()
    input_uncertainty_prior = prior_params.loc["input_uncertainty"].to_list()
    obs_uncertainty_prior = prior_params.loc["obs_uncertainty"].to_list()

    # calculate and plot prior distribution
    x_k, pdf_k = normal_pdf(k_prior[0], k_prior[1])
    axes[0, 0].plot(x_k, pdf_k, "orange")
    x_initial_state, pdf_initial_state = normal_pdf(
        initial_state_prior[0], initial_state_prior[1]
    )
    axes[1, 1].plot(x_initial_state, pdf_initial_state, "orange")
    x_input_uncertainty, pdf_input_uncertainty = normal_pdf(
        input_uncertainty_prior[0], input_uncertainty_prior[1]
    )
    axes[2, 2].plot(x_input_uncertainty, pdf_input_uncertainty, "orange")
    x_obs_uncertainty, pdf_obs_uncertainty = normal_pdf(
        obs_uncertainty_prior[0], obs_uncertainty_prior[1]
    )
    axes[3, 3].plot(x_obs_uncertainty, pdf_obs_uncertainty, "orange")

    # plot posterior distribution
    sns.kdeplot(plot_df["k"], ax=axes[0, 0], color="C0")
    sns.kdeplot(plot_df["initial state"], ax=axes[1, 1], color="C0")
    sns.kdeplot(plot_df["input uncertainty"], ax=axes[2, 2], color="C0")
    sns.kdeplot(plot_df["obs uncertainty"], ax=axes[3, 3], color="C0")

    # plot true parameters
    axes[1, 0].plot([k_true], [initial_state_true], "r*", markersize=10)
    axes[2, 0].plot([k_true], [input_uncertainty_true], "r*", markersize=10)
    axes[3, 0].plot([k_true], [obs_uncertainty_true], "r*", markersize=10)
    axes[2, 1].plot([initial_state_true], [input_uncertainty_true], "r*", markersize=10)
    axes[3, 1].plot([initial_state_true], [obs_uncertainty_true], "r*", markersize=10)
    axes[3, 2].plot(
        [input_uncertainty_true], [obs_uncertainty_true], "r*", markersize=10
    )

    # plot parameter contours
    sns.kdeplot(
        x=plot_df["k"],
        y=plot_df["initial state"],
        cmap="Blues",
        shade=True,
        levels=levels,
        ax=axes[1, 0],
    )
    sns.kdeplot(
        x=plot_df["k"],
        y=plot_df["input uncertainty"],
        cmap="Blues",
        shade=True,
        levels=levels,
        ax=axes[2, 0],
    )
    sns.kdeplot(
        x=plot_df["k"],
        y=plot_df["obs uncertainty"],
        cmap="Blues",
        shade=True,
        levels=levels,
        ax=axes[3, 0],
    )
    sns.kdeplot(
        x=plot_df["initial state"],
        y=plot_df["input uncertainty"],
        cmap="Blues",
        shade=True,
        levels=levels,
        ax=axes[2, 1],
    )
    sns.kdeplot(
        x=plot_df["initial state"],
        y=plot_df["obs uncertainty"],
        cmap="Blues",
        shade=True,
        levels=levels,
        ax=axes[3, 1],
    )

    sns.kdeplot(
        x=plot_df["input uncertainty"],
        y=plot_df["obs uncertainty"],
        cmap="Blues",
        shade=True,
        levels=levels,
        ax=axes[3, 2],
    )

    # clean up labels
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")
    axes[2, 2].set_xlabel("")
    axes[2, 2].set_ylabel("")
    axes[3, 3].set_ylabel("")
    axes[1, 0].set_xlabel("")
    axes[2, 0].set_xlabel("")
    axes[2, 1].set_xlabel("")
    axes[2, 1].set_ylabel("")
    axes[3, 1].set_ylabel("")
    axes[3, 2].set_ylabel("")
    # axes[0, 0].set_ylabel("pdf", fontsize=label_fontsize)
    # axes[1, 1].set_ylabel("pdf", fontsize=label_fontsize)
    # axes[2, 2].set_ylabel("pdf", fontsize=label_fontsize)
    # axes[3, 3].set_ylabel("pdf", fontsize=label_fontsize)
    axes[0, 0].set_ylabel("k", fontsize=label_fontsize)
    axes[1, 0].set_ylabel("Initial state", fontsize=label_fontsize)
    axes[2, 0].set_ylabel("Input uncertainty", fontsize=label_fontsize)
    axes[3, 0].set_ylabel("Output uncertainty", fontsize=label_fontsize)
    axes[3, 0].set_xlabel("k", fontsize=label_fontsize)
    axes[3, 1].set_xlabel("Initial state", fontsize=label_fontsize)
    axes[3, 2].set_xlabel("Input uncertainty", fontsize=label_fontsize)
    axes[3, 3].set_xlabel("Output uncertainty", fontsize=label_fontsize)

    # set same scale
    axes[1, 0].set_ylim(axes[1, 1].get_xlim())
    axes[2, 0].set_ylim(axes[2, 2].get_xlim())
    axes[2, 1].set_ylim(axes[2, 2].get_xlim())
    axes[3, 0].set_ylim(axes[3, 3].get_xlim())
    axes[3, 1].set_ylim(axes[3, 3].get_xlim())
    axes[3, 2].set_ylim(axes[3, 3].get_xlim())

    axes[1, 0].set_xlim(axes[0, 0].get_xlim())
    axes[2, 0].set_xlim(axes[0, 0].get_xlim())
    axes[3, 0].set_xlim(axes[0, 0].get_xlim())
    axes[2, 1].set_xlim(axes[1, 1].get_xlim())
    axes[3, 1].set_xlim(axes[1, 1].get_xlim())
    axes[3, 2].set_xlim(axes[2, 2].get_xlim())

    # remove unnecessary ticks
    axes[0, 0].set_xticks([])
    axes[1, 0].set_xticks([])
    axes[1, 1].set_xticks([])
    axes[2, 0].set_xticks([])
    axes[2, 1].set_xticks([])
    axes[2, 2].set_xticks([])

    axes[2, 1].set_yticks([])
    axes[3, 1].set_yticks([])
    axes[3, 2].set_yticks([])

    # adjust labels
    legend_labels = [
        "Ground truth",
        "Posterior distribution",
        "Prior distribution",
    ]
    legend_marker_symbol = ["*", "_", "_"]
    legend_marker_color = ["r", "blue", "orange"]
    for i in range(3):
        axes[0, 0].scatter(
            [],
            [],
            color=legend_marker_color[i],
            marker=legend_marker_symbol[i],
            label=legend_labels[i],
            s=200,
        )
    fig.legend(frameon=False, fontsize=18, bbox_to_anchor=(0.7, 0.6), loc="center")

    fig.suptitle(
        f"Posterior distribution of parameters",
        x=0.6,
        y=0.85,
        fontsize=20,
    )

    fig.subplots_adjust(wspace=0.26, hspace=0.1, top=0.9, right=0.9)

    return fig


# %%
def plot_scenarios(
    truth_df: pd.DataFrame,
    estimation: pd.DataFrame,
    start_ind: int,
    stn_i: int,
    sig_q: float,
    line_mode: Optional[bool] = False,
):
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    # uncertainty bounds
    ax[0].fill_between(
        truth_df["index"][1:],
        truth_df["J_true"][1:] - 1.96 * 0.02 / stn_i,
        truth_df["J_true"][1:] + 1.96 * 0.02 / stn_i,
        color="grey",
        alpha=0.3,
        label=r"95% theoretical uncertainty bounds",
    )

    # scenarios
    if line_mode:
        alpha = 5./estimation["input"][start_ind:,].shape[0]
        ax[0].plot(
            truth_df["index"][1:],
            estimation["input"][start_ind:, 1:].T,
            color="C9",
            linewidth=1.5,
            zorder=0,
            alpha=alpha,
        )

    else:
        sns.boxplot(
            estimation["input"][start_ind:,],
            orient="v",
            ax=ax[0],
            color="C9",
            linewidth=1.5,
            fliersize=1.5,
            zorder=0,
            width=0.8,
            fill=False,
            whis=(2.5, 97.5),
        )
        ax[0].plot(
            truth_df["index"][1:],
            estimation["input"][start_ind:,1:].mean(axis=0),
            color="C9",
            linewidth=3,
            zorder=0,
            label=f"Ensemble mean"
        )

    # truth trajectory
    ax[0].plot(
        truth_df["index"][1:], truth_df["J_true"][1:], color="k", label="Truth", linewidth=0.8
    )
    # observations
    ax[0].scatter(
        truth_df["index"][truth_df["is_obs"]][1:],
        truth_df["J_obs"][truth_df["is_obs"]][1:],
        marker="+",
        c="k",
        s=100,
        linewidth=2,
        label="Observations",
    )

    ax[0].set_ylim([max(truth_df["J_true"]) * 1.065, min(truth_df["J_true"]) * 0.925])
    # ax[0].set_xlim([0, len(truth_df["index"])])
    ax[0].set_ylabel("Input signal", fontsize=16)
    ax[0].set_xticks([])

    # ========================================
    # uncertainty bounds

    ax[1].fill_between(
        truth_df["index"][1:],
        truth_df["Q_true"][1:] - 1.96 * sig_q,
        truth_df["Q_true"][1:] + 1.96 * sig_q,
        color="grey",
        alpha=0.3,
        label=r"95% theoretical uncertainty",
    )
    # scenarios
    if line_mode:
        ax[1].plot(
            truth_df["index"][1:],
            estimation["output"][start_ind:, 1:].T,
            color="C9",
            linewidth=1.5,
            zorder=0,
            alpha=alpha,
        )
    else:
        sns.boxplot(
            estimation["output"][start_ind:,1:],
            order=truth_df["index"]-1,
            orient="v",
            ax=ax[1],
            color="C9",
            linewidth=1.5,
            fliersize=1.5,
            zorder=0,
            width=0.8,
            fill=False,
            whis=(2.5, 97.5),
        )
        ax[1].plot(
            truth_df["index"][1:],
            estimation["output"][start_ind:, 1:].mean(axis=0),
            color="C9",
            linewidth=3,
            zorder=0,
            label=f"Ensemble mean"
        )

    # truth trajectory
    ax[1].plot(
        truth_df["index"][1:], truth_df["Q_true"][1:], color="k", label="Truth", linewidth=0.8
    )
    # observations
    ax[1].scatter(
        truth_df["index"][truth_df["is_obs"]][1:],
        truth_df["Q_true"][truth_df["is_obs"]][1:],
        marker="+",
        c="k",
        s=100,
        linewidth=2,
        label="Observations",
    )
    cyan_line, = ax[1].plot([], [], 'c-', label='Scenarios')

    ax[1].set_ylim([min(truth_df["Q_true"]) * 0.9972, max(truth_df["Q_true"]) * 1.0028])
    # ax[1].set_xlim([0, len(truth_df["index"])])
    ax[1].set_ylabel("Output signal", fontsize=16)
    ax[1].set_xlabel("Timestep", fontsize=16)
    if line_mode:
        ax[1].legend(frameon=False, ncol = 5, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    else:
        ax[1].legend(frameon=False, ncol = 4, loc="upper center", bbox_to_anchor=(0.5, 1.1))

    fig.tight_layout()
    return fig


# %%
def convergence_check_plot(plot_df: pd.DataFrame, plot_end_ind: int) -> plt.Figure:
    plot_df = (plot_df - plot_df.mean(axis=0)) / plot_df.std(axis=0, ddof=1)
    plot_df = plot_df.iloc[:plot_end_ind]
    plot_df.plot()
    plt.legend(frameon=False)
    plt.ylabel("Standardized parameter values")
    plt.xlabel("MCMC iteration")
    plt.show()
    return
