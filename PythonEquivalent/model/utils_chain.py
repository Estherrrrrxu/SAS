# %%
from model.model_interface import ModelInterface
import numpy as np
import scipy.stats as ss
from dataclasses import dataclass
from functions.utils import _inverse_pmf
# %%
@dataclass
class State:
    """ Class for keep track of state at each timestep"""
    R: np.ndarray  # [T, N], input scenarios
    W: np.ndarray  # [N], log weight at each observed timestep
    X: np.ndarray  # [T+1, N], state at each timestep
    A: np.ndarray  # [T+1, N], ancestor at each timestep

class Chain:
    def __init__(self,
        model_interface: ModelInterface,
        theta: np.ndarray
    ) -> None:
        """
        Args:
            model_interface (ModelInterface): model interface object -- assume this is already initialized
            theta (np.ndarray): theta that has been generated from parameter space
        """
        # TODO: think about this more
        self.model_interface = model_interface
        self.model_interface.update_model(theta)
        self.model_interface.input_model()
        self.R = self.model_interface.R
        self._state_init = self.model_interface.initial_state

        # get dimension constants
        self.N = self.model_interface.N
        self.T = self.model_interface.T
        self.K = self.model_interface.K

        # get observation
        self.outflux = self.model_interface.outflux

        # initialize state
        A = np.zeros((self.N, self.K + 1)).astype(int)
        A[:,0] = np.arange(self.N)
        self.state_init = State( 
            R=self.R,
            W=np.log(np.ones(self.N) / self.N),
            X=np.ones((self.N, self.T + 1)) * self._state_init,
            A=A
        )


    def run_sequential_monte_carlo(self) -> None:
        """Run sequential Monte Carlo
        """
        R = self.state_init.R
        W = self.state_init.W
        X = self.state_init.X
        A = self.state_init.A

        # TODO: work on diff k and T later
        for k in range(self.K):
            # draw new state samples and associated weights based on last ancestor
            xk = X[A[:,k],k]
            # TODO: may need more more work on this cuz currently only one flux
            xkp1 = self.model_interface.transition_model(Xtm1=xk,
                                                         Rt=R[A[:,k],k])

            wkp1 = W + np.log(
                self.model_interface.observation_model(Xk=xkp1,
                                                       yt=self.outflux[k])
                            )
            W = wkp1
            X[:,k+1] = xkp1
            A[:,k+1] = _inverse_pmf(A[:,k],W, num = self.N)

        self.state = State(X=X, A=A, W=W, R=R)  
        
        # generate new set of R
        self.model_interface.input_model()
        self.R = self.model_interface.R    
        return 

    def run_particle_MCMC(self) -> None:
        """Run particle MCMC
        """
        R = self.state_init.R
        W = self.state_init.W
        X = self.state_init.X
        A = self.state_init.A

        # sample an ancestral path based on final weight
        B = self._find_traj(A, W)
        # reinitialize weight
        W = np.log(np.ones(self.N)/self.N)

        for k in range(self.K):
            rr = R[:,k]
            xkp1 = self.model_interface.transition_model(Xtm1=X[:,k],
                                                         Rt=R[:,k])
            # Update 
            x_prime = X[B[k+1],k+1]

            W_tilde = W + np.log(
                self.model_interface.state_model(x_prime=x_prime, xkp1=xkp1)
            )
            
            A[B[k+1],k+1] = _inverse_pmf(xkp1 - x_prime,
                                         W_tilde, 
                                         num = 1)
            # now update everything in new state
            notB = np.arange(0,self.N)!=B[k+1]
            A[:,k+1][notB] = _inverse_pmf(X[:,k],W, num = self.N-1)
            xkp1[notB] = xkp1[A[:,k+1][notB]]
            rr[notB] = rr[A[:,k+1][notB]]
            xkp1[~notB] = xkp1[A[B[k+1],k+1]]
            rr[~notB] = R[:,k][A[B[k+1],k+1]]
            X[:,k+1] = xkp1   
            R[:,k] = rr       
            W[notB] = W[A[:,k+1][notB]]
            W[~notB] = W[A[B[k+1],k+1]]
            wkp1 = W + np.log(
                        self.model_interface.observation_model(
                                                            Xk=xkp1,
                                                            yt=self.outflux[k])
                            )
            W = wkp1#/wkp1.sum()
        
        self.state = State(X = X, A = A, W = W, R = R)  
        # generate new set of R
        self.model_interface.input_model()
        self.R = self.model_interface.R           
        return

    def _find_traj(self, A: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Find particle trajectory based on final weight
        
        Args:
            A (np.ndarray): Ancestor matrix
            W (np.ndarray): Weight at final timestep
        
        Returns:
            np.ndarray: Trajectory indices of reference particle
        """


        B = np.zeros(self.K+1).astype(int)
        B[-1] = _inverse_pmf(A[:,-1],W, num = 1)
        for i in reversed(range(1,self.K+1)):
            B[i-1] =  A[:,i][B[i]]
        return B

    def _get_X_traj(self, X:  np.ndarray, B: np.ndarray) -> np.ndarray:
        """Get X trajectory based on sampled particle trajectory
        
        Args:
            X (np.ndarray): State variable
            B (np.ndarray): Trajectory indices of reference particle

        Returns:
            np.ndarray: Trajectory of X that is sampled at final timestep
        """

        traj_X = np.zeros(self.K+1)
        for i in range(self.K+1):
            traj_X[i] = X[B[i],i]
        return traj_X
    
    def _get_R_traj(self, R: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Get R trajectory based on sampled particle trajectory

        Args:
            R (np.ndarray): Observation variable
            B (np.ndarray): Trajectory indices of reference particle
        
        Returns:
            np.ndarray: Trajectory of R that is sampled at final timestep
        """
        traj_R = np.zeros(self.K)
        for i in range(self.K):
            traj_R[i] = R[B[i+1],i]
        return  traj_R

# %%
