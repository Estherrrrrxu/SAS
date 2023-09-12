# %%
# Create dataset for linear reservoir model given various possible input scenarios
# 1. white noise
# 2. exponential decay
# 3. real precipitation data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.stats as ss

from typing import List, Optional
from copy import deepcopy
import os
# %%
def linear_reservoir_transition_model(
        qt: float,
        k: float,
        delta_t: float,
        jt: float
    ) -> float:
    """Give four inputs about the watershed at timestep t, return the calculated discharge at t+1
    Args:
        qt (float): discharge at timestep t
        k (float): linear reservoir factor
        delta_t (float): time interval for precip
        jt (float): input precipitation at timestep t
    Returns:
        float: discharge at timestep t+1

    """
    qtp1 = (1 - k * delta_t) * qt + delta_t * jt
    return qtp1

def generate_inflow_white_noise(
    mean: float,
    std: float,
    length: int
    ) -> List[float]:
    """ Generate white noise as inflow with given mean and std
        Note that all inflow values are positive
    Args:
        mean (float): mean of the white noise
        std (float): standard deviation of the white noise
        length (int): length of the time series
    Returns:
        List[float]: inflow time series
    """
    # chop precipitation data
    J = np.random.normal(mean, std, length)
    J[J<0] = 0
    return J

def generate_inflow_exp_decay(
    theta: float,
    length: int
    ) -> List[float]:
    """ Generate inflow with exponential decay
        theta: decay rate
        length: length of the time series
    """
    # chop precipitation data
    J = np.random.exponential(theta, length)
    return J


# make function
def generate_outflow(
    J: List[float],
    delta_t: float, 
    k: float, 
    Q_init:float
    ) -> List[float]:
    """Use transition model to generate outflow given input precipitation
    Args:
        J (List[float]): input precipitation time series
        delta_t (float): time interval for precip
        k (float): linear reservoir factor
        length (int): length of the time series
        Q_init (float): intial discharge, default = 0.  
    Returns:
        List[float]: discharge time series    
    """
    # calculate discharge using transition model
    length = len(J)
    Q = np.zeros(length+1)
    Q[0] = Q_init
    for i in range(length):
        Q[i+1] = linear_reservoir_transition_model(Q[i], k, delta_t, J[i])
    return Q[1:]

# %%
def generate_w_diff_noise_level(
        root: str,
        stn_ipt: float,
        stn_obs: float,
        params_ipt: Optional[List[float]]=None,
        input_precip: List[float]=None,
        type_ipt: str="WhiteNoise",
        length: int=100,
        Q_init: float=0.,
        k: float=1.,
        delta_t: float=1./24/60*15
    ) -> None:
    """Generate dataset with different noise level
    Args:
        root (str): root directory to save the dataset
        stn_ipt (float): signal to noise ratio for input precipitation
        stn_obs (float): signal to noise ratio for observed discharge
        params_ips (Optional[List[float]], optional): parameters for input precipitation. Defaults to None.
        input_precip (List[float], optional): input precipitation time series. Defaults to None.
        type_ipt (str, optional): type of input precipitation. Defaults to "WhiteNoise".
        length (int, optional): length of the time series. Defaults to 100.
        Q_init (float, optional): initial discharge. Defaults to 0..
        k (float, optional): linear reservoir factor. Defaults to 1..
    """

    # generate inflow
    if type_ipt == "WhiteNoise" and params_ips is not None:
        J = generate_inflow_white_noise(params_ipt[0], params_ipt[1], length)
    elif type_ipt == "ExpDecay" and params_ipt is not None:
        J = generate_inflow_exp_decay(params_ipt, length)
    elif type_ipt == "RealPrecip" and input_precip is not None:
        J = input_precip
    else:
        raise ValueError("Check input type and required parameters")
    
    # generate outflow
    Q = generate_outflow(J, delta_t, k, Q_init)
    e = J * delta_t
    sig_e = np.std(e, ddof=1)
    phi = 1 - k * delta_t
    sig_q = np.sqrt(sig_e ** 2/(1 - phi ** 2))

    J_true = deepcopy(J)
    Q_true = deepcopy(Q)
    
    # add noise according to signal to noise ratio
    noise_j = sig_e/delta_t/stn_ipt
    noise_q = sig_q/stn_obs

    a = (0 - Q)/noise_q
    Q = ss.truncnorm.rvs(a, np.inf, loc=Q, scale=noise_q)
    a = (0 - J)/noise_j
    J = ss.truncnorm.rvs(a, np.inf, loc=J, scale=noise_j)

    df = pd.DataFrame({'J_true': J_true,'J_obs': J, 'Q_true': Q_true, 'Q_obs': Q})

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(df['J_true'], label="Truth")
    plt.plot(df['J_obs'], "*", label="Obs")
    plt.legend(frameon=False)
    plt.title('Input')

    plt.subplot(2,1,2)
    plt.plot(df['Q_true'], label="Truth")
    plt.plot(df['Q_obs'], "*", label="Obs")
    plt.title('Output')
    plt.legend(frameon=False)
    plt.tight_layout()
    if not os.path.exists(root + f"{type_ipt}/"):
        os.mkdir(root + f"{type_ipt}/")

    plt.savefig(root + f"{type_ipt}/stn_{stn_ipt}_{stn_obs}.pdf")
    df.to_csv(root + f"{type_ipt}/stn_{stn_ipt}_{snt_obs}.csv")
    return


# %%
if __name__ == "__main__":
    root = "/Users/esthersida/Documents/Code/particle/SAS/PythonEquivalent/Data/"
    # universal constants
    length = 1000
    delta_t = 1./24/60*15
    k = 1.
    phi = 1 - k * delta_t
    # for input and output stn ratio are consistent
    for stn_ipt in [1, 2, 3, 4, 5]:
        for ratio in [0.5, 1, 2, 3, 4, 5, 6]:
            # white noise
            params_ips = [0.5, 0.02]
            snt_obs = stn_ipt * ratio
            generate_w_diff_noise_level(root, stn_ipt, snt_obs, params_ips, type_ipt="WhiteNoise", length=length, k=k, delta_t=delta_t, Q_init=0.5)

            # exponential decay
            params_ips = 0.5
            generate_w_diff_noise_level(root, stn_ipt, snt_obs, params_ips, type_ipt="ExpDecay", length=length, k=k, delta_t=delta_t, Q_init=0.5)

            # real precipitation
            precip = pd.read_csv(root + "precip.csv")
            input_precip = precip['45'].to_numpy()
            input_precip = input_precip[900:900+length]

            generate_w_diff_noise_level(root, stn_ipt, snt_obs, input_precip=input_precip, type_ipt="RealPrecip", length=length, k=k, delta_t=delta_t, Q_init=0.01)

# %%
