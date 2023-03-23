# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from functions.models import transition_model
import scipy.stats as ss
# %%
# ==========================
# processes
# ==========================

def input_model(J:float, theta:dict, N:int):
    """
        Generate random uniform noise
    """
    theta_val = theta['not_to_estimate']['input_uncertainty']
    return ss.uniform(J-theta_val,theta_val).rvs(N)

def observation_model(xht:List[float],theta_val:float,xt:List[float]):
    return ss.norm(xht,theta_val).pdf(xt)

def transition_model(qt: float,k: float,delta_t: float,jt: float):
    """
        give four inputs about the watershed at timestep t
        return the calculated discharge at t+1
    """
    qtp1 = (1 - k * delta_t) * qt + k * delta_t * jt
    return qtp1
