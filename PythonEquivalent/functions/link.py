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
class ModelLink:
    """
    A class used to specify inputs and settings for SSModel (State Space Model)

    Attributes
    ----------
    df : pd.DataFrame
        A dataframe contains all essential information as the input 
    config : dict
        A dictionary contains information about parameters to estimate and not to be estimated
    influx : List[float]
        The influx of the model
    outflux : List[fleat]
        The outflux of the model

    Methods
    -------
    input_model(influx, theta, N)
        Generate the input time series for given period

    transition_model()
        Generate the state to the next set of time

    observation_model()
        Generate observation at given time
        
    """
    def __init__(self, ):



    def input_model(self, J:float, theta:dict, N:int):
        """
            Generate random uniform noise
        """
        theta_val = theta['not_to_estimate']['input_uncertainty']
        return ss.uniform(J-theta_val,theta_val).rvs(N)

    def observation_model(self, xht:List[float],theta_val:float,xt:List[float]):
        return ss.norm(xht,theta_val).pdf(xt)

    def transition_model(self, qt: float,k: float,delta_t: float,jt: float):
        """
            give four inputs about the watershed at timestep t
            return the calculated discharge at t+1
        """
        qtp1 = (1 - k * delta_t) * qt + k * delta_t * jt
        return qtp1
