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
def data_generation(
    delta_t: float, 
    k: float, 
    mean: float, 
    std: float, 
    length: int,
    Q_init:float
    ):
    """
        precip: input precip time series
        delta_t: time interval for precip
        mean
    """
    # chop precipitation data
    J = np.random.normal(mean, std, length)
    J[J<0] = 0


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
        theta_obs: float,
        mean: float = -0.01,
        std: float = 0.02,
        length: int = 100,
        Q_init: float = 0.004,
        name: str = "WN"
    ) -> None:
    delta_t = 1./24/60*15
    k = 1.
    #
    J,Q = data_generation(delta_t, k, mean, std, length, Q_init)
    Q_true = Q.copy()
    J_true = J.copy()
    # pretend we know obs error

    a = (0 - Q)/theta_obs
    Q = ss.truncnorm.rvs(a, np.inf,loc = Q,scale = theta_obs)
    a = (0 - J)/theta_ipt
    J = ss.truncnorm.rvs(a, np.inf,loc = J,scale = theta_ipt)

    df = pd.DataFrame({'J_true': J_true,'J_obs': J, 'Q_true': Q_true, 'Q_obs': Q})

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
    plt.savefig(root + f"{name}_ipt_{theta_ipt}_obs_{theta_obs}.pdf")

    df.to_csv(root + f"{name}_ipt_{theta_ipt}_obs_{theta_obs}.csv")
# %%
if __name__ == "__main__":
    root = "/Users/esthersida/Documents/Code/particle/SAS/PythonEquivalent/Data/"
    theta_ipt = 0.002
    theta_obs = 0.00006
    # TODO: need to think more on this



# %%
