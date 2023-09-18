# %%
from model.model_interface import ModelInterface
import numpy as np
import scipy.stats as ss
from dataclasses import dataclass
from functions.utils import _inverse_pmf
from copy import deepcopy
from typing import Optional
# %%
@dataclass
class State:
    """ Class for keep track of state at each timestep"""
    R: np.ndarray  # [N, T], input scenarios
    W: np.ndarray  # [N], log weight at each observed timestep
    X: np.ndarray  # [N, T], state at each timestep
    A: np.ndarray  # [N, K], ancestor at each timestep
    Y: np.ndarray  # [N, T], observation at each timestep
    
class Chain:
    def __init__(self,
        model_interface: ModelInterface,
        theta: Optional[np.ndarray] = None
    ) -> None:
        """
        Args:
            model_interface (ModelInterface): model interface object -- assume this is already initialized
            theta (np.ndarray): theta that has been generated from parameter space
        """
        # TODO: think about this more
        self.model_interface = model_interface
        self.model_interface.update_model(theta_new=theta)
        self.R = self.model_interface.R

        # get dimension constants
        self.N = self.model_interface.N
        self.T = self.model_interface.T
        self.K = self.model_interface.K
        # get observed indices
        self.observed_ind = self.model_interface.observed_ind
        # get observation
        self.outflux = self.model_interface.outflux

        # initialize state
        self._update_state_init()

        t_k_map = np.arange(0,self.T+1)[self.observed_ind]
        # if there are no gap in data, do nothing
        post_ind = t_k_map + 1
        if self.K == self.T:
            pre_ind = t_k_map
        # if the first one is observed:
        else:
            pre_ind = t_k_map - 1
            pre_ind[0] = t_k_map[0]

        self.pre_ind = pre_ind
        self.post_ind = post_ind

    #subject to change
    def _update_state_init(self) -> None:
        A = np.zeros((self.N, self.K)).astype(int)
        A[:,0] = np.arange(self.N)
        self.state_init = State( 
            R=self.R,
            W=np.ones(self.N) / self.N,
            X=np.ones((self.N, self.T)),     
            Y=np.zeros((self.N, self.T)),
            A=A
        )

    def run_sequential_monte_carlo(self) -> None:
        """Run sequential Monte Carlo
        """
        R = self.state_init.R
        W = self.state_init.W
        X = self.state_init.X
        A = self.state_init.A
        Y = self.state_init.Y

        # initialization at the first observation
        ind_init = self.pre_ind[0]
        X[:,ind_init] = X[:,ind_init] * self.model_interface.theta.initial_state
        Y[:,ind_init] = self.model_interface.observation_model(Xk=X[:,ind_init])
        W_init = np.log(self.model_interface.observation_model_likelihood(
                                                        yhk=Y[:,ind_init],
                                                        yk=self.outflux[ind_init]
                                                        ))
        W_init = np.exp(W_init - W_init.max())
        W_init *= W
        W = W_init/W_init.sum()

        for k in range(1, self.K):
            A[:,k] = _inverse_pmf(A[:,k-1], W, num = self.N)
            
            start_ind = self.pre_ind[k]
            end_ind = self.post_ind[k]

            Rt = self.model_interface.input_model(start_ind=start_ind, end_ind=end_ind)
            
            # the state at the last time step of the previous interval
            xk = X[A[:,k], self.post_ind[k-1]-1]

            # TODO: may need more more work on this cuz currently only one flux
            xkp1 = self.model_interface.transition_model(Xtm1=xk, Rt=Rt)    

            Y[:,start_ind:end_ind] = self.model_interface.observation_model(Xk=xkp1)

            w_temp = np.log(self.model_interface.observation_model_likelihood(
                                                        yhk=Y[:,end_ind-1],
                                                        yk=self.outflux[end_ind-1]
                                                        ))
            
            w_temp = np.exp(w_temp - w_temp.max())
            W *= w_temp
            W /= W.sum()

            # This p1 takes account for initial condition
            X[:,start_ind:end_ind] = xkp1

        self.state = State(X=X, A=A, W=W, R=R, Y=Y)  

        

        return 
    

    def run_particle_MCMC_AS(self) -> None:
        """Run particle MCMC with Ancestor Sampling (AS)
        """
        # updated states from previous run
        R = self.state.R
        W = self.state.W
        X = self.state.X
        A = self.state.A
        Y = self.state.Y
        
        # sample an ancestral reference trajectory based on final weight
        B = self._find_traj(A, W)
        notB = np.arange(0,self.N) != B[0]

        ind_init = self.pre_ind[0]
        # resample states that are not reference trajectory
        X[notB, ind_init] = X[notB, ind_init] * self.model_interface.theta.initial_state[notB] 
        # TODO: check if this initial state is generated according to new distribution
        Y[:,ind_init] = self.model_interface.observation_model(Xk=X[:,ind_init])
        W_init = np.log(self.model_interface.observation_model_likelihood(
                                                        yhk=Y[:,ind_init],
                                                        yk=self.outflux[ind_init]
                                                        ))
        W_init = np.exp(W_init - W_init.max())
        W_init *= W
        W = W_init/W_init.sum()

        for k in range(1, self.K):

            start_ind_k = self.pre_ind[k]
            end_ind_k = self.post_ind[k]
         
            # propagate the all particles
            xkm1 = X[:, start_ind_k-1] # the last one in the previous interval
            Rk = self.model_interface.input_model(start_ind=start_ind_k, end_ind=end_ind_k)
            xk = self.model_interface.transition_model(Xtm1=xkm1,
                                                         Rt=Rk)
            
            # referenc  e traj in this trajectory
            x_prime = X[B[k], start_ind_k: end_ind_k]
            offset = xk - x_prime
            sd = offset.mean(axis=0)

            W_tilde = sum(
                [np.log(self.model_interface.state_as_model(x_prime=x_prime[i],
                                                            xkp1=xk[:,i], 
                                                            sd=abs(sd))) for i in range(len(x_prime))]
                )
            
            W_tilde = np.exp(W_tilde - W_tilde.max())
            W_tilde /= W_tilde.sum()
            print("============= W tilde =============")
            print(W_tilde)

            # resample ancestor
            A[B[k-1],k] = _inverse_pmf(offset, W_tilde, num = 1)
            notB = np.arange(0,self.N) != B[k-1]

            A[:,k][notB] = _inverse_pmf(X[:,k], W, num = self.N-1)

            xk[notB] = xk[A[:,k][notB]]
            Rk[notB] = Rk[A[:,k][notB]]
            xk[~notB] = xk[A[B[k-1],k]]
            Rk[~notB] = R[:,k][A[B[k-1],k]]

            # update state
            X[:,start_ind_k:end_ind_k] = xk
            R[:,start_ind_k:end_ind_k] = Rk

            # update weight       
            W[notB] = W[A[:,k][notB]]
            W[~notB] = W[A[B[k-1],k]]
            Y[:,start_ind_k:end_ind_k] = self.model_interface.observation_model(Xk=xk)

            wkp1 = np.log(self.model_interface.observation_model_likelihood(
                                                        yhk=Y[:,end_ind_k-1],
                                                        yk=self.outflux[end_ind_k-1]
                                                        ))  
            wkp1 = np.exp(wkp1 - wkp1.max())
            W = wkp1/wkp1.sum()
            print(W)
             
        self.state = State(X=X, A=A, W=W, R=R, Y=Y)       
        return

    def _find_traj(self, A: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Find particle trajectory based on final weight
        
        Args:
            A (np.ndarray): Ancestor matrix
            W (np.ndarray): Weight at final timestep
        
        Returns:
            np.ndarray: Trajectory indices of reference particle
        """

        B = np.zeros(self.K).astype(int)
        B[-1] = _inverse_pmf(A[:, -1], W, num=1)
        for i in reversed(range(1, self.K)):
            B[i-1] = A[:,i][B[i]]

        return B

    def _get_X_traj(self, X: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Get X trajectory based on sampled particle trajectory
        
        Args:
            X (np.ndarray): State variable
            B (np.ndarray): Trajectory indices of reference particle

        Returns:
            np.ndarray: Trajectory of X that is sampled at final timestep
        """
        traj_X = np.ones(self.T) * self.model_interface.theta.initial_state
        for i in range(self.K):
            traj_X[self.pre_ind[i]+1:self.post_ind[i]+1] = X[B[i],self.pre_ind[i]+1:self.post_ind[i]+1]
        return traj_X
    
    def _get_Y_traj(self, Y:  np.ndarray, B: np.ndarray) -> np.ndarray:
        """Get Y trajectory based on sampled particle trajectory
        
        Args:
            Y (np.ndarray): Predicted observation
            B (np.ndarray): Trajectory indices of reference particle

        Returns:
            np.ndarray: Trajectory of X that is sampled at final timestep
        """
        traj_Y = np.zeros(self.T)

        for i in range(self.K):
            traj_Y[self.pre_ind[i]:self.post_ind[i]] = Y[B[i+1],self.pre_ind[i]:self.post_ind[i]]

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
        
        for i in range(self.K-1):
            traj_R[self.pre_ind[i]:self.post_ind[i]] = R[B[i+1],self.pre_ind[i]:self.post_ind[i]]
        traj_R[self.pre_ind[-1]:] = R[B[-1],self.pre_ind[-1]:]
        return traj_R

# %%
