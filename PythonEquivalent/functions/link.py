# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import scipy.stats as ss
from estimator import *

# %%
class LinearReservoirInputModel(InputModel):
    """
    Args:
        theta (np.ndarray): [U_upper, N]
    """
    def input(self, Ut:float) -> np.ndarray:
        """Input model for linear reservoir

        Rt = Ut - U(0, U_upper)

        Args:
            Ut (float): forcing at time t

        Returns:
            np.ndarray: Rt
        """
        return ss.uniform(Ut-self.theta[0],self.theta[0]).rvs(self.theta[1])

class LinearReservoirTranModel(TransitionModel):
    """
    Args:
        theta (np.ndarray): [k, delta_t]
    """
    def transition(self, xtm1: np.ndarray, rt: float) -> np.ndarray:
        """Transition model for linear reservoir

        xt = (1 - k * delta_t) * x_{t-1} + k * delta_t * rt
        
        Args:
            xtm1 (np.ndarray): state at t-1
            rt (float): uncertainty r at t

        Returns:
            np.ndarray: xt
        """
        xt = (1 - self.theta[0] * self.theta[1]) * xtm1 + self.theta[0] * self.theta[1] * rt
        return xt
    
class LinearReservoirObsModel(ObservationModel):
    def observation(self, xk: np.ndarray) -> np.ndarray:
        """Observation model for linear reservoir

        y_hat(k) = x(k)

        Args:
            xk (np.ndarray): observed y at time k

        Returns:
            np.ndarray: y_hat
        """
        return xk
    
class LinearReservoirProposalModel(ProposalModel):
    """_summary_

    Args:
        theta (np.array): sig_v
    """
    def f_theta(self, xtm1: np.ndarray, ut: np.ndarray) -> np.ndarray:
        """Call transition_model directly

        Args:
            xtm1 (np.ndarray): state at t-1
            rt (float): uncertainty r at t

        Returns:
            np.ndarray: xt
        """
        return self.transition_model.transition(xtm1, ut)

    def g_theta(self, yht: np.ndarray, yt: np.ndarray) -> np.ndarray:
        """Observation model for linear reservoir

        y(t) = y_hat(t) + N(0, theta_v)

        Args:
            yht (np.ndarray): estimated y_hat at time t
            yt (np.ndarray): observed y at time t
            theta (float): parameter of the model

        Returns:
            np.ndarray: p(y|y_hat, sig_v)
        """
        return ss.norm(yht, self.theta).pdf(yt)
    







my_ss_model = SSModel(
    transition_model= ,
    proposal_model=,
    observation_model=LinearReservoirObsModel(theta = 1.),
)



my_ss_model.run_sequential_monte_carlo(influx, )
..








