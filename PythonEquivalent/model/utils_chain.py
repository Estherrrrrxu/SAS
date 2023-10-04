# %%
from model.model_interface import ModelInterface
import numpy as np
import scipy.stats as ss
from dataclasses import dataclass
from functions.utils import _inverse_pmf
from typing import Optional



# %%
@dataclass
class State:
    """Class for keep track of state at each timestep"""

    R: np.ndarray  # [N, T], input scenarios
    W: np.ndarray  # [N], weight at each observed timestep
    X: np.ndarray  # [N, T], state at each timestep
    A: np.ndarray  # [N, K], ancestor at each timestep
    Y: np.ndarray  # [N, T], observation at each timestep

class Chain:
    def __init__(
        self, model_interface: ModelInterface, theta: Optional[np.ndarray] = None
    ) -> None:
        """
        Args:
            model_interface (ModelInterface): model interface object -- assume this is already initialized
            theta (np.ndarray): theta that has been generated from parameter space
        """
        # TODO: think about this more
        self.model_interface = model_interface
        self.model_interface.update_theta(theta_new=theta)

        # get dimension constants
        self.N = self.model_interface.N
        self.T = self.model_interface.T
        self.K = self.model_interface.K

        # get observed indices
        self.observed_ind = self.model_interface.observed_ind
        # get observation
        self.outflux = self.model_interface.outflux

        # initialize observation indices
        self._set_observation_indices()

    def _set_observation_indices(self) -> None:
        """Set indices for observation"""
        t_k_map = np.arange(0, self.T + 1)[self.observed_ind]
        # if there are no gap in data, do nothing
        post_ind = t_k_map + 1
        if self.K == self.T:
            pre_ind = t_k_map
        # other wise, fill in the gap
        else:
            pre_ind = t_k_map - 1
            pre_ind[0] = t_k_map[0]

        self.pre_ind = pre_ind
        self.post_ind = post_ind

    def run_sequential_monte_carlo(self) -> None:
        """Run sequential Monte Carlo"""

        # initialize variables
        R=np.ones((self.N, self.T))
        W=np.ones(self.N) / self.N
        X=np.ones((self.N, self.T))
        Y=np.ones((self.N, self.T))
        A=np.zeros((self.N, self.K)).astype(int)
        A[:, 0] = Ak = np.arange(self.N)

        # initialization at the first observation
        ind_init = self.pre_ind[0]
        X[:, ind_init] = X[:, ind_init] * self.model_interface.initial_state_model(num=self.N)
        Y[:, ind_init] = self.model_interface.observation_model(Xk=X[:, ind_init])
        W_init = self.model_interface.observation_model_probability(
                yhk=Y[:, ind_init], yk=self.outflux[ind_init]
            )
        W_init = np.exp(W_init - W_init.max())
        W = W_init / W_init.sum()

        # start sequential Monte Carlo for each observation
        for k in range(1, self.K):
            start_ind_k = self.pre_ind[k]
            end_ind_k = self.post_ind[k]

            Rt = self.model_interface.input_model(start_ind=start_ind_k, end_ind=end_ind_k)

            # the state at the last time step of the previous interval
            Ak = _inverse_pmf(X[Ak, start_ind_k - 1], W, num=self.N)
            xkm1 = X[Ak, start_ind_k - 1]

            # propagate particles
            xk = self.model_interface.transition_model(Xtm1=xkm1, Rt=Rt)
            yk = self.model_interface.observation_model(Xk=xk)

            w_temp = self.model_interface.observation_model_probability(
                    yhk=yk[:, -1], yk=self.outflux[end_ind_k - 1]
                )

            w_temp = np.exp(w_temp - w_temp.max())
            W = w_temp / w_temp.sum()

            # This p1 takes account for initial condition
            X[:, start_ind_k:end_ind_k] = xk
            R[:, start_ind_k:end_ind_k] = Rt
            Y[:, start_ind_k:end_ind_k] = yk
            A[:, k] = Ak

        self.state = State(X=X, A=A, W=W, R=R, Y=Y)

        return
    
    def run_particle_MCMC_AS(self) -> None:
        """Run particle MCMC with Ancestor Sampling (AS)"""

        # inherit states from previous run
        R = self.state.R
        W = self.state.W
        X = self.state.X
        A = self.state.A
        Y = self.state.Y

        # sample an ancestral reference trajectory based on final weight
        B = self._find_traj(A, W)
        nB = np.arange(self.N) != B[0]


        ind_init = self.pre_ind[0]
        # resample states that are not reference trajectory
        X[nB, ind_init] = self.model_interface.initial_state_model(num=self.N - 1)
        Y[:, ind_init] = self.model_interface.observation_model(Xk=X[:, ind_init])

        W_init = self.model_interface.observation_model_probability(
                yhk=Y[:, ind_init], yk=self.outflux[ind_init]
            )
        W_init = np.exp(W_init - W_init.max())
        W = W_init / W_init.sum()

        # start particle MCMC for each observation
        for k in range(1, self.K):
            start_ind_k = self.pre_ind[k]
            end_ind_k = self.post_ind[k]

            # propagate the all particles
            xkm1 = X[:, start_ind_k - 1]  # the last one in the previous interval
            Rk = self.model_interface.input_model(
                start_ind=start_ind_k, end_ind=end_ind_k
            )
            xk = self.model_interface.transition_model(Xtm1=xkm1, Rt=Rk)

            # reference traj in this trajectory
            Bk = B[k]
            x_prime = X[Bk, end_ind_k - 1]
            offset = xk[:, -1] - x_prime
            sd = offset.mean()

            W_tilde = self.model_interface.state_as_probability(offset=offset, std=abs(sd)/3.0)
            W_tilde = np.exp(W_tilde - W_tilde.max())
            W_tilde /= W_tilde.sum()

            # resample to get particle indices that propagate from k-1 to k
            Bkm1 = B[k - 1]
            nB = np.arange(self.N) != Bkm1
            A[nB, k] = _inverse_pmf(xkm1, W, num=self.N - 1)
            A[Bkm1, k] = _inverse_pmf(offset, W_tilde, num=1)
            # or get MAP
            # A[B[k-1],k] = np.argmax(W_tilde)

            # now only retain xk that are propogated from k-1 to k
            nBk = np.arange(self.N) != Bk
            xk[nBk] = xk[A[nB, k]]
            xk[Bk] = X[Bk, start_ind_k:end_ind_k] # reference trajectory does not change

            Rk = Rk[A[:, k]]
            Rk[Bk] = R[Bk, start_ind_k:end_ind_k] # scenario does not change for future yet

            # update weight
            yk = self.model_interface.observation_model(Xk=xk)
            wkp1 = self.model_interface.observation_model_probability(
                    yhk=yk[:,-1], yk=self.outflux[end_ind_k - 1]
                )
            wkp1 = np.exp(wkp1 - wkp1.max())
            W = wkp1 / wkp1.sum()

            # updates everything
            X[:, start_ind_k:end_ind_k] = xk
            R[:, start_ind_k:end_ind_k] = Rk
            Y[:, start_ind_k:end_ind_k] = yk

        self.state = State(X=X, A=A, W=W, R=R, Y=Y)
        return

    def _find_traj(self, A: np.ndarray, W: np.ndarray, max: Optional[bool] = False) -> np.ndarray:
        """Find particle trajectory based on final weight

        Args:
            A (np.ndarray): Ancestor matrix
            W (np.ndarray): Weight at final timestep

        Returns:
            np.ndarray: Trajectory indices of reference particle
        """

        B = np.zeros(self.K).astype(int)
        if max:
            B[-1] = np.argmax(W)
        else:
            B[-1] = _inverse_pmf(A[:, -1], W, num=1)
            
        for i in reversed(range(1, self.K)):
            B[i - 1] = A[:, i][B[i]]

        return B

    def _get_X_traj(self, X: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Get X trajectory based on sampled particle trajectory

        Args:
            X (np.ndarray): State variable
            B (np.ndarray): Trajectory indices of reference particle

        Returns:
            np.ndarray: Trajectory of X that is sampled at final timestep
        """
        traj_X = np.ones(self.T)
        for i in range(self.K):
            traj_X[self.pre_ind[i] : self.post_ind[i]] = X[
                B[i], self.pre_ind[i] : self.post_ind[i]
            ]
        return traj_X

    def _get_Y_traj(self, Y: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Get Y trajectory based on sampled particle trajectory

        Args:
            Y (np.ndarray): Predicted observation
            B (np.ndarray): Trajectory indices of reference particle

        Returns:
            np.ndarray: Trajectory of X that is sampled at final timestep
        """
        traj_Y = np.zeros(self.T)

        for i in range(self.K):
            traj_Y[self.pre_ind[i] : self.post_ind[i]] = Y[
                B[i], self.pre_ind[i] : self.post_ind[i]
            ]
        return traj_Y

    def _get_R_traj(self, R: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Get R trajectory based on sampled particle trajectory

        Args:
            R (np.ndarray): Observation variable
            B (np.ndarray): Trajectory indices of reference particle

        Returns:
            np.ndarray: Trajectory of R that is sampled at final timestep
        """

        traj_R = np.zeros(self.T)

        for i in range(self.K):
            traj_R[self.pre_ind[i] : self.post_ind[i]] = R[
                B[i], self.pre_ind[i] : self.post_ind[i]
            ]
        return traj_R


# %%
