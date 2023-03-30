from dataclasses import dataclass
from abc import ABC
import numpy as np
import pandas as pd


@dataclass
class State:
    R: np.ndarray  # [len, part]
    W: np.ndarray  # [part]
    X: np.ndarray  # [len+1, part]
    A: np.ndarray  # [len+1, part]


class ObservationModel(ABC):
    def observe(s: State, t: float) -> np.ndarray:
        ...


class TransitionModel(ABC):

    def __init__(self, k: float):
        self.k = k
    
    def transition(last_state: State, t: float, dt: float) -> State:
        ...


class ProposalModel(ABC):
    def f_theta():
        ...

    def g_theta():
        ...


class SSModel(ABC):
    
    def __init__(
        self,
        N: int,
        K: int,
        observation_model: ObservationModel,
        transition_model: TransitionModel,
        proposal_model: ProposalModel
    ):
        self.N = N
        self.K = K
        self.observation_model = observation_model
        self.transition_model = transition_model
        self.proposal_model = proposal_model
    
    def run_sequential_monte_carlo(
        influx: pd.Series,
        outflux: pd.Series,
        theta_to_estimate: Tuple[float, float]
    ) -> State:
        
        T = len(influx)
        R = np.zeros((self.N, T))
        W = np.log(np.ones(self.N) / self.N)

        X = np.zeros((self.N, self.K + 1))
        X[:, 0] = np.ones(self.N) * outflux[0]

        A = np.zeros((self.N, self.K + 1)).astype(int)
        A[:, 0] = np.arange(self.N)

        for kk in range(self.K):
            ...
    
    def run_particle_gibbs_sampling(
        self,
        state: State,
    ) -> State:
        
        State(R=.., X=.., A=.., W=..)


