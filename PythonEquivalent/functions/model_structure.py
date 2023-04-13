# %%
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
import scipy.stats as ss
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC


# %%
@dataclass
class Parameter:
    input_model: np.ndarray
    transition_model: np.ndarray
    observation_model: np.ndarray

class SpecifyModel(ABC):
    """General model structure that can be customized in ModelLink

    Args:
        theta_input (np.ndarray): parameters of input model
        theta_transition (np.ndarray): parameters of transition model
        theta_observation (np.ndarray): parameters of observation model

        Ut (np.ndarray): forcing at time t
        Xtm1 (np.ndarray): state at t-1
        Rt (np.ndarray): uncertainty r at t
        Xk (np.ndarray): observed x at observation time k

    Methods:
        input_model: generate input uncertainty
        transition: specify transition model
        observation_model: specify observation model
    
    """
    def __init__(
            self, 
            theta: Parameter,
            N: int
        ) -> None:
        """Initialize model parameters 

        Args:
            theta (np.ndarray): parameter of the model
            N (int): number of particles
        """
        # need theta be decoded in this way
        self.theta_input = theta.input_model
        self.theta_transition = theta.transition_model
        self.theta_observation = theta.observation_model
        self.N = N
    
    def input_model(
            self, 
            ut: np.ndarray
        ) -> np.ndarray:
        """Generate input uncertainty

        Args:
            ut (np.ndarray): forcing at time t
            self.theta_input (np.ndarray): parameters of input model
        """
        ...
  
    def transition(
            self, 
            xtm1: np.ndarray, 
            rt: np.ndarray
        ) -> np.ndarray:
        """Specify transition model

        Args:
            xtm1 (np.ndarray): state at t-1
            rt (float): uncertainty r at t
            self.theta_transition (np.ndarray): parameters of transition model
        """
        ...

    def observation_model(self, xk: np.ndarray) -> np.ndarray:
        """Specify observation model

        Args:
            xk (np.ndarray): observed x at time k
            self.theta_observation (np.ndarray): parameters of observation model
        """
        ...



class LinearReservior(SpecifyModel):
    def input_model(self, Ut:float) -> np.ndarray:
        """Input model for linear reservoir

        Rt = Ut - U(0, U_upper)

        Args:
            Ut (float): forcing at time t

        Returns:
            np.ndarray: Rt
        """
        return ss.uniform(Ut-self.theta_input, 
                          self.theta_input
                          ).rvs(self.N)

    def transition_model(self, xtm1: np.ndarray, rt: float) -> np.ndarray:
        """Transition model for linear reservoir

        xt = (1 - k * delta_t) * x_{t-1} + k * delta_t * rt
        
        Args:
            xtm1 (np.ndarray): state at t-1
            rt (float): uncertainty r at t

        Returns:
            np.ndarray: xt
        """
        theta = self.theta_transition.values()
        theta_k = theta[0]
        theta_dt = theta[1]
        xt = (1 -  theta_k * theta_dt) * xtm1 + theta_k * theta_dt * rt
        return xt
    
    def observation_model(self, xk: np.ndarray) -> np.ndarray:
        """Observation model for linear reservoir

        y_hat(k) = x(k)

        Args:
            xk (np.ndarray): observed y at time k

        Returns:
            np.ndarray: y_hat
        """
        return xk






    

