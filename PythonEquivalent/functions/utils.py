# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
def _inverse_pmf(x: np.ndarray, pmf: np.ndarray, num: int) -> np.ndarray:
    """Sample x based on its ln(pmf) using discrete inverse sampling method

    Args:
        x (np.ndarray): The specific values of x
        pmf (np.ndarray): The weight (pmf) associated with x
        num (int): The total number of samples to generate

    Returns:
        np.ndarray: index of x that are been sampled according to its ln(pmf)
    """
    # Sort x and pmf together based on x
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    pmf_sorted = pmf[sorted_indices]

    # Compute the cumulative sum of the sorted PMF
    cumulative_pmf = pmf_sorted.cumsum()

    # Generate uniform random values
    u = np.random.uniform(size=num)

    # Use searchsorted to find the indices
    ind_sample = np.searchsorted(cumulative_pmf, u)

    # Ensure the indices do not exceed the maximum index
    ind_sample[ind_sample == len(cumulative_pmf)] -= 1

    # Map the sampled indices back to the original order
    original_indices = sorted_indices[ind_sample]

    return original_indices


# use df['index'] to make the actual plot
def plot_base(df_obs):
    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    ax[0].bar(df_obs["index"], df_obs["J_true"], width=1, color="k", label="Truth")
    ax[0].plot(
        df_obs["index"][df_obs["is_obs"] == True],
        df_obs["J_obs"][df_obs["is_obs"] == True],
        "+",
        color="C3",
        markersize=7,
        label="Observation",
    )
    ax[0].set_ylabel("")
    ax[0].set_ylim([max(df_obs["J_true"])*1.05, max(df_obs["J_true"])*0.75])
    ax[0].legend(frameon=False, ncol=2)
    ax[0].set_xticks([])
    ax[0].set_title("Input")

    ax[1].plot(df_obs["index"], df_obs["Q_true"], "k.-", label="Truth")
    ax[1].plot(
        df_obs["index"][df_obs["is_obs"] == True],
        df_obs["Q_obs"][df_obs["is_obs"] == True],
        "+",
        color="C3",
        markersize=7,
        label="Observation",
    )
    ax[1].set_ylabel("")
    ax[1].set_xlabel("Time [day]")
    ax[1].legend(frameon=False)
    ax[1].set_title("Output")
    return fig, ax


def plot_bulk(weekly_bulk):
    fig, ax = plot_base(weekly_bulk)
    temp = weekly_bulk[weekly_bulk["is_obs"] == True]
    ax[0].plot(
        temp["index"],
        temp["J_obs"],
        marker="o",
        markerfacecolor="none",
        markeredgecolor="green",
        linestyle="",
        label="Observed Time Stamp",
    )
    ax[0].plot(
        weekly_bulk["index"],
        weekly_bulk["J_obs"],
        "+",
        color="C3",
        markersize=7,
    )
    ax[0].legend(frameon=False, ncol=3)
    ax[0].set_ylim([max(weekly_bulk["J_true"]) * 1.05, max(weekly_bulk["J_true"]) * 0.75])

    ax[1].plot(
        temp["index"],
        temp["Q_obs"],
        marker="o",
        markerfacecolor="none",
        markeredgecolor="green",
        linestyle="",
        label="Observed Time Stamp",
    )
    ax[1].legend(frameon=False, ncol=3)
    return fig, ax


def plot_MAP(
    state,
    df: pd.DataFrame,
    pre_ind: int,
    post_ind: int,
    left: int = None,
    right: int = None,
):
    """Plot the MAP trajectory

    Args:
        state (State): State object
        df (pd.DataFrame): Dataframe of observations
        pre_ind (int): Index of the first observation
        post_ind (int): Index of the last observation
        left (int, optional): Left bound of the plot. Defaults to None.
        right (int, optional): Right bound of the plot. Defaults to None.

    Returns:
        np.ndarray: MAP trajectory
    """

    # %%
    # state = chain.state
    # df = case.df_obs
    # pre_ind = chain.pre_ind
    # post_ind = chain.post_ind
    # left, right = None, None
    X = state.X
    A = state.A
    W = state.W
    R = state.R
    Y = state.Y
    K = A.shape[1]

    if left is None:
        left = df["index"][df["is_obs"]==True].values[0]
    if right is None:
        right = df["index"][df["is_obs"]==True].values[-1]

    B = np.zeros(K).astype(int)
    B[-1] = _inverse_pmf(A[:, -1], W, num=1)
    for i in reversed(range(1, K)):
        B[i - 1] = A[:, i][B[i]]

    T = len(X[0])
    MAP = np.zeros(T)
    MAP_R = np.zeros(T)

    for i in range(K):
        MAP[pre_ind[i] : post_ind[i]] = Y[B[i], pre_ind[i] : post_ind[i]]

    for i in range(K):
        MAP_R[pre_ind[i] : post_ind[i]] = R[B[i], pre_ind[i] : post_ind[i]]

    fig, ax = plot_base(df)
    ax[0].plot(
        df["index"].values,
        MAP_R,
        "s",
        color="C9",
        markersize=7,
        mfc="none",
        label="One Traj/MLE",
    )
    ax[0].set_xlim([left, right])

    ax[1].plot(
        df["index"].values,
        MAP,
        "|",
        markersize=15,
        markeredgewidth=1.5,
        linestyle=(0, (3, 1)),
        color="C9",
        linewidth=2,
        label="One Traj/MLE",
    )
    ax[1].set_xlim([left, right])
    ax[1].set_ylim([min(df["Q_true"])*0.85, max(df["Q_true"]) * 1.05])

    return fig, ax

# %%
def create_bulk_sample(original: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Create bulk sample from original data.

    Parameters
    ----------
    original : pd.DataFrame
        Original data.
    n : int
        Number of observations to be grouped.

    Returns
    -------
    pd.DataFrame
        Bulk sample.
    """

    is_obs = (original.index) % n == 0
    last_true_index = np.where(is_obs)[0][-1]
    ind_group = (original.index-1) // n

    bulk = (
        original.iloc[: last_true_index + 1]
        .groupby(ind_group[: last_true_index + 1])
        .mean()
    )
    bulk["index"] = original.loc[is_obs, "index"].values

    df_temp = pd.merge(original, bulk, on="index", how="left")
    df_temp.columns = ['J_true_fine', 'J_obs_fine', 'Q_true_fine', 'Q_obs_fine', 'index', 'is_obs_fine', 'J_true', 'J_obs', 'Q_true', 'Q_obs', 'is_obs']
    df_temp = df_temp.drop(
        ["is_obs_fine"], axis=1
    )

    bulk = df_temp.fillna(method="bfill").dropna()
    bulk["is_obs"] = is_obs[: last_true_index + 1]
    return bulk


# %%
def normalize_over_interval(
    arr: np.ndarray, input: float
):
    """Normalize the values of the array over interval before making observation

    Args:
        arr (np.ndarray): Generated array
        input (float): Input array that generated array should be similar to

    """
    # sum all array values
    sum_val = sum(arr)
    if sum_val == 0:
        raise ValueError("Sum of subarray is 0")
    multiplier = input / sum_val * len(arr)

    return arr * multiplier


# %%
