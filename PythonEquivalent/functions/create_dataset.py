import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.stats as ss

from typing import List

# %%
def create_nonlinear_drift_time_series(
        T: int = 100, 
        theta_true: List[float] = [-0.1, 0.3], 
        dt: float = 0.01
    ) -> pd.DataFrame:
    """Create input time series
    
    Consider the following general example:
        dX_t = (theta_true[0] X_t + cos(\pi X_t)) dt + X_t dW_t
    where X_t is the state variable and W_t is the Wiener process.
    The observation is given by:
        Y_t = X_t + N(0, theta_true[1])

    Args:
        T (int, optional): Length of time series. Defaults to 100.
        theta_true (List[float, float], optional): Parameters of the model. Defaults to [-0.1, 0.3].
        dt (float, optional): Time step. Defaults to 0.01.

    Returns:
        pd.DataFrame: Dataframe with columns X (state, truth) and Y (observation)
    """
    X = np.zeros(T)
    W = np.zeros(T)
    dw = ss.norm.rvs(size=T-1)
    W[1:] = np.cumsum(dw)

    for k in range(1, T):
        X[k] = X[k-1] + (theta_true[0] * X[k-1] + np.cos(np.pi * X[k-1])) * dt + X[k-1] * dw[k-1]
    
    Y = X + ss.norm(0, theta_true[1]).rvs(size=T)

    plt.figure(figsize=(10,5))
    plt.plot(X, label = 'State')
    plt.plot(Y, '*', label = 'Observation')
    plt.legend()
    plt.ylabel('Value')
    plt.xlabel('Time')
    plt.show()

    # create a dataframe
    df = pd.DataFrame(columns=['X', 'Y'])
    df['X'] = X.reshape(-1)
    df['Y'] = Y.reshape(-1)
    df.to_csv('Nonlinear_Drift.csv')
    return df

df = create_nonlinear_drift_time_series()
# %%
def transition_model(qt: float,k: float,delta_t: float,jt: float):
    """
        give four inputs about the watershed at timestep t
        return the calculated discharge at t+1
    """
    qtp1 = (1 - k * delta_t) * qt + k * delta_t * jt
    return qtp1

# make function
def data_generation(precip: List[float], \
    delta_t: float, k: float, l: int, Q_init = 1.):
    """
        precip: input precip time series
        delta_t: time interval for precip
        k: linear reservoir factor
        l: chop the precipitation time series to N/l
        Q_init: intial discharge, default = 0. 
    """
    # chop precipitation data
    length = round(len(precip)/l)
    J = precip[:length]
    # calculate discharge using transition model
    Q = np.zeros(length+1)
    Q[0] = Q_init
    for i in range(length):
        Q[i+1] = transition_model(Q[i], k, delta_t, J[i])
    df = pd.DataFrame({"J":J,"Q":Q[1:]})
    df = df.dropna()
    return df['J'].values, df['Q'].values

precip = pd.read_csv("precip.csv", index_col = 0)
precip = precip['45'].values
# define constant
l = 100 # control the length of the time series
delta_t = 1./24/60*15
k = 1.
#
J,Q = data_generation(precip, delta_t, k , l)
Q_true = Q.copy()
J_true = J.copy()
# pretend we know obs error
theta_ipt = 0.254*delta_t
theta_obs = 0.00005
Q += np.random.normal(0, theta_obs,size = len(Q))
J += np.random.uniform(0, theta_ipt,size = len(Q))
df = pd.DataFrame({'J_true': J_true,'J_obs': J, 'Q_true': Q_true, 'Q_obs': Q})
df = df[600:]
df = df.reset_index()

plt.figure()
plt.subplot(2,1,1)
plt.plot(df['J_obs'],"*",label = "J obs")
plt.plot(df['J_true'],label = "J true")
plt.legend()
plt.title('J')
plt.subplot(2,1,2)
plt.plot(df['Q_obs'],label = "Q obs")
plt.plot(df['Q_true'],label = "Q true")
plt.title('Q')
plt.legend()
plt.tight_layout()
df.to_csv("Dataset.csv")
# %%