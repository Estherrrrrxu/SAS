# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%
def _inverse_pmf(x: np.ndarray,ln_pmf: np.ndarray, num: int) -> np.ndarray:
    """Sample x based on its ln(pmf) using discrete inverse sampling method

    Args:
        x (np.ndarray): The specific values of x
        pmf (np.ndarray): The weight (ln(pmf)) associated with x
        num (int): The total number of samples to generate

    Returns:
        np.ndarray: index of x that are been sampled according to its ln(pmf)
    """
    ind = np.argsort(x) # sort x according to its magnitude
    pmf = np.exp(ln_pmf) # convert ln(pmf) to pmf
    pmf /= pmf.sum()
    pmf = pmf[ind] # sort pdf accordingly
    u = np.random.uniform(size = num)
    ind_sample = np.searchsorted(pmf.cumsum(), u)
    # cannot exceed the maximum index
    ind_sample[ind_sample == len(pmf)] -= 1
 
    return ind[ind_sample]


# use df['index'] to make the actual plot
def plot_base(df, df_obs):
    fig, ax = plt.subplots(2, 1, figsize=(8,5))
    ax[0].bar(df['index'], df['J_true'], 
            width = 1, color = 'k', label = 'Truth')
    ax[0].plot(df_obs['index'], df_obs['J_obs'], 
               '+', color='C3', markersize=7,label='Observation')
    ax[0].invert_yaxis()
    ax[0].set_ylabel("Precipitation [mm]")
    ax[0].legend(frameon = False)
    ax[0].set_xticks([])
    ax[0].set_title("Preciptation")

    ax[1].plot(df['index'], df['Q_true'], 
               color = 'k', label = 'Truth')    
    ax[1].plot(df_obs['index'], df_obs['Q_obs'], 
               '+', color = 'C3', markersize=7,
               label = 'Observation')
    ax[1].set_ylabel("Discharge [mm]")
    ax[1].set_xlabel("Time [day]")
    ax[1].legend(frameon = False)
    ax[1].set_title("Discharge")
    return fig, ax

def plot_bulk(original, weekly_bulk):
    fig, ax = plot_base(original, weekly_bulk)
    temp = weekly_bulk[weekly_bulk['is_obs']]
    ax[0].plot(temp['index'], temp["J_obs"], marker='o', markerfacecolor='none', markeredgecolor='green', linestyle='',label='Observed Time Stamp')
    ax[0].legend(frameon=False)
    ax[1].plot(temp['index'], temp["Q_obs"], marker='o', markerfacecolor='none', markeredgecolor='green', linestyle='',label='Observed Time Stamp')
    ax[1].legend(frameon=False)
    return fig, ax

def plot_MLE(state, df, df_obs: pd.DataFrame, pre_ind, post_ind,
              left: int=None, right: int=None):
    """make the plot appropriately to present model output

    Args:
        state (_type_): _description_
        df (_type_): _description_
        left (int, optional): _description_. Defaults to 0.
        right (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    """
    # state = chain.state
    # df=case.df
    # df_obs=case.df_obs
    # pre_ind = chain.pre_ind
    # post_ind = chain.post_ind
    # left,right = None,None
    X = state.X
    A = state.A
    W = state.W
    R = state.R
    Y = state.Y

    J_obs = df['J_obs'].values
    Q_obs = df['Q_obs'].values
    K = A.shape[1]-1

    if left is None:
        left = df_obs['index'].values[0]
    if right is None:
        right = df_obs['index'].values[-1]

    B = np.zeros(K+1).astype(int)
    B[-1] = _inverse_pmf(W,A[:,-1], num = 1)
    for i in reversed(range(1,K+1)):
        B[i-1] =  A[:,i][B[i]]

    T = len(X[0])-1
    MLE = np.zeros(T+1)
    MLE_R = np.zeros(T)

    for i in range(K):
        MLE[pre_ind[i]:post_ind[i]] = X[B[i+1],pre_ind[i]+1:post_ind[i]+1]
        # MLE[i] = Y[B[i+1],i]
    for i in range(K):
        MLE_R[pre_ind[i]:post_ind[i]] = R[B[i+1],pre_ind[i]:post_ind[i]]
        # MLE_R[i] = R[B[i+1],i]   


    fig, ax = plot_base(df,df_obs)
    ax[0].plot(df_obs['index'].values, MLE_R,
               "s", color='C9', markersize=7, mfc='none',
               label = "One Traj/MLE")
    ax[0].set_xlim([left,right])

    ax[1].plot(df_obs['index'].values, MLE[:-1], "|", markersize=15,markeredgewidth=1.5,
               linestyle=(0, (3, 1)), color='C9', linewidth=2, 
               label = "One Traj/MLE")
    ax[1].set_xlim([left,right])
    return MLE
# %%
def plot_scenarios(df, df_obs, model, start_ind):
    fig, ax = plt.subplots(2, 1, figsize=(8,5))
    ax[0].bar(df['index'], df['J_true'], 
            width = 1, color = 'k', label = 'Truth')
    ax[0].plot(df_obs['index'].values, model.input_record.T[:,start_ind:],
            "s", color='C9', markersize=7, mfc='none')
    ax[0].plot(df_obs['index'].values, model.input_record.T[:,-1],
            "s", color='C9', markersize=7, mfc='none', label = 'Input scenarios')
    ax[0].plot(df_obs['index'], df_obs['J_obs'], 
               '+', color='C3', markersize=7,label='Observation')
    ax[0].invert_yaxis()
    ax[0].set_ylabel("Precipitation [mm]")
    ax[0].legend(frameon = False)
    ax[0].set_xticks([])
    ax[0].set_title("Preciptation")

    ax[1].plot(df['index'], df['Q_true'], 
               color = 'k', label = 'Truth')   
    ax[1].plot(df_obs['index'].values, model.output_record.T[:,start_ind:-1],
            linestyle=(1, (1, 1)), color='C9', linewidth=2)
    ax[1].plot(df_obs['index'].values, model.output_record.T[:,-1],
            linestyle=(1, (1, 1)), color='C9', linewidth=2, label = 'Trajectories') 
    ax[1].plot(df_obs['index'], df_obs['Q_obs'], 
               '+', color = 'C3', markersize=7,
               label = 'Observation')
    ax[1].set_ylabel("Discharge [mm]")
    ax[1].set_xlabel("Time [day]")
    ax[1].legend(frameon = False)
    ax[1].set_title("Discharge")
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

    is_obs = (original.index + 1) % n == 0
    last_true_index = np.where(is_obs)[0][-1]
    ind_group = original.index // n

    bulk = original.iloc[:last_true_index + 1].groupby(ind_group[:last_true_index + 1]).mean()
    bulk['index'] = original.loc[is_obs, 'index'].values

    df_temp = pd.merge(original, bulk, on='index', how='left')
    df_temp = df_temp.drop(df_temp.columns[1:5], axis=1)
    df_temp.columns = original.columns
    df_temp['Q_true'] = original['Q_true']
    df_temp['J_true'] = original['J_true']

    bulk = df_temp.fillna(method='bfill').dropna()
    bulk['is_obs'] = is_obs[:last_true_index + 1]
    return bulk

# %%
def normalize_over_interval(
        arr: np.ndarray, 
        index_array: np.ndarray, 
        input: np.ndarray
    ):
    """Normalize the values of the array over interval before making observation

    Args:
        arr (np.ndarray): Generated array
        index_array (np.ndarray): Array of observed indices
        input (np.ndarray): Input array that generated array should be similar to
    
    """
    normalized_arr = arr.copy()
    post_ind = index_array + 1
    pre_ind = np.concatenate(([0], post_ind[:-1]))
    # Normalize each interval separately
    for i in range(len(index_array)):
        start_index = pre_ind[i]
        end_index = post_ind[i]
        # Extract the subarray between the specified indices
        subarray = arr[start_index:end_index]

        # Find the minimum and maximum values within the subarray
        sum_val = sum(subarray)
        multiplier = [x / sum_val for x in subarray]

        target_val = input[end_index-1]
        normalized_subarray = [x * target_val * len(subarray) for x in multiplier]

        # Replace the original subarray with the normalized values
        normalized_arr[start_index:end_index] = normalized_subarray

    return normalized_arr
# %%
