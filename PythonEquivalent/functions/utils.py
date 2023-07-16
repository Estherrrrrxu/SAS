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

def plot_MLE(state, df, df_obs: pd.DataFrame,
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
    X = state.X
    A = state.A
    W = state.W
    R = state.R
    Y = state.Y

    J_obs = df['J_obs'].values
    Q_obs = df['Q_obs'].values
    K = len(J_obs)

    if left is None:
        left = df_obs['index'].values[0]
    if right is None:
        right = df_obs['index'].values[-1]

    B = np.zeros(K+1).astype(int)
    B[-1] = _inverse_pmf(W,A[:,-1], num = 1)
    for i in reversed(range(1,K+1)):
        B[i-1] =  A[:,i][B[i]]
    MLE = np.zeros(K+1)
    MLE_R = np.zeros(K)
    for i in range(K):
        MLE[i] = Y[B[i+1],i]
    for i in range(K):
        MLE_R[i] = R[B[i+1],i]   
    T = len(X[0])-1

    fig, ax = plot_base(df,df_obs)
    ax[0].plot(df_obs['index'].values, MLE_R,
               "s", color='C9', markersize=7, mfc='none',
               label = "One Traj/MLE")
    ax[0].set_xlim([left,right])

    ax[1].plot(df_obs['index'].values, MLE[:-1],
               linestyle=(0, (3, 1)), color='C9', linewidth=2, 
               label = "One Traj/MLE")
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
