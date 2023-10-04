# %%
# This is the plotting script for:
# 1. the (joint) posterior distribution of parameters
# 2. the prediction and scenarios of the model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from typing import List, Optional
from numba import vectorize


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
        "Truth/Truth approximation",
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
    truth_df: pd.DataFrame, estimation: pd.DataFrame, start_ind: int, stn_i: int, sig_q: float
):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    # uncertainty bounds
    ax[0].fill_between(
        truth_df["index"],
        truth_df["J_true"] - 2 * 0.02 / stn_i,
        truth_df["J_true"] + 2 * 0.02 / stn_i,
        color="grey",
        alpha=0.3,
        label=r"95% theoretical uncertainty bounds",
    )

    # scenarios
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
    )

    # truth trajectory
    ax[0].plot(
        truth_df["index"], truth_df["J_true"], color="k", label="Truth", linewidth=0.8
    )
    # observations
    ax[0].scatter(
        truth_df["index"][truth_df['is_obs']],
        truth_df["J_obs"][truth_df['is_obs']],
        marker="+",
        c="k",
        s=100,
        linewidth=2,
        label="Observations",
    )

    ax[0].set_ylim([max(truth_df["J_true"]) * 1.05, min(truth_df["J_true"]) * 0.95])
    ax[0].set_ylabel("Precipitation [mm]")
    ax[0].legend(frameon=False)
    ax[0].set_xticks([])
    ax[0].set_title("Input")

    # ========================================
    # uncertainty bounds

    ax[1].fill_between(
        truth_df["index"],
        truth_df["Q_true"] - 2 * sig_q,
        truth_df["Q_true"] + 2 * sig_q,
        color="grey",
        alpha=0.3,
        label=r"95% theoretical uncertainty bounds",
    )
    # scenarios
    sns.boxplot(
        estimation["output"][start_ind:,],
        orient="v",
        ax=ax[1],
        color="C9",
        linewidth=1.5,
        fliersize=1.5,
        zorder=0,
        width=0.8,
        fill=False,
    )

    # truth trajectory
    ax[1].plot(
        truth_df["index"], truth_df["Q_true"], color="k", label="Truth", linewidth=0.8
    )
    # observations
    ax[1].scatter(
        truth_df["index"][truth_df['is_obs']],
        truth_df["Q_true"][truth_df['is_obs']],
        marker="+",
        c="k",
        s=100,
        linewidth=2,
        label="Observations",
    )

    ax[1].set_ylim([min(truth_df["Q_true"]) * 0.998, max(truth_df["Q_true"]) * 1.002])
    ax[1].set_ylabel("Discharge [mm]")
    ax[1].set_xlabel("Timestep [15 min]")
    ax[1].legend(frameon=False)
    ax[1].set_title("Output")

    fig.tight_layout()
    return fig
 
# %%
if __name__ == "__main__":
    # load data
    N, D, L = 30, 30, 10
    stn_i = 5
    path_str = f"../Results/TestLR/WhiteNoise/{stn_i}_N_{N}_D_{D}_L_{L}"

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

    k_true = 1.0
    initial_state_true = 0.5001100239495894
    input_uncertainty_true = 0.02 / stn_i
    obs_uncertainty_true = 0.0
    dt = 1.0 / 24 / 60 * 15

    sig_e = dt * 0.02 / stn_i
    phi = 1 - k_true * dt
    sig_q = np.sqrt(sig_e**2 / (1 - phi**2))

    true_params = [
        k_true,
        initial_state_true,
        input_uncertainty_true,
        obs_uncertainty_true,
        sig_q
    ]

    prior_params = pd.read_csv(f"{path_str[:-17]}/prior_{stn_i}.csv", index_col=0)

    # plot posterior
    g = plot_parameter_posterior(plot_df, true_params, prior_params, threshold)
    g.savefig(f"{path_str}/posterior.pdf")

    # plot trajectories
    estimation = {"input": input_scenarios, "output": output_scenarios}
    truth_df = pd.read_csv(f"{path_str}/df_ipt.csv", index_col=0)

    g = plot_scenarios(truth_df, estimation, threshold, stn_i, sig_q)
    g.savefig(f"{path_str}/scenarios.pdf")

    # calculate KL divergence
    true_q_mean = truth_df["Q_true"].to_numpy()
    true_q_std = sig_q
    obs_q_mean = output_scenarios[threshold:,].mean(axis=0)
    obs_q_std = output_scenarios[threshold:,].std(axis=0)

    KL = cal_KL(true_q_mean, obs_q_mean, true_q_std, obs_q_std)
    plt.figure()
    plt.plot(KL)
    plt.show()



# %%
