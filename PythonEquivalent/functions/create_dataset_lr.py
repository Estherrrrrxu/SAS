# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.stats as ss

from typing import List

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

# %%
def generate_w_diff_noise_level(
        root: str,
        theta_ipt: float, 
        theta_obs: float
    ) -> None:
    precip = pd.read_csv(root + "precip.csv", index_col = 0)
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

    a = (0 - Q)/theta_obs
    Q = ss.truncnorm.rvs(a, np.inf,loc = Q,scale = theta_obs)
    a = (0 - J)/theta_ipt
    J = ss.truncnorm.rvs(a, np.inf,loc = J,scale = theta_ipt)

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
    plt.savefig(root + f"LR_ipt_{theta_ipt}_obs_{theta_obs}.pdf")

    df.to_csv(root + f"LR_ipt_{theta_ipt}_obs_{theta_obs}.csv")
# %%
if __name__ == "__main__":
    root = "/Users/esthersida/Documents/Code/particle/SAS/PythonEquivalent/Data/"
    theta_ipt = 0.0005
    theta_obs = 0.00005
    generate_w_diff_noise_level(root, theta_ipt, theta_obs)
    theta_ipt = 0.005
    theta_obs = 0.0005
    generate_w_diff_noise_level(root, theta_ipt, theta_obs)
    theta_ipt = 0.005
    theta_obs = 0.0001
    generate_w_diff_noise_level(root, theta_ipt, theta_obs)

# %%
