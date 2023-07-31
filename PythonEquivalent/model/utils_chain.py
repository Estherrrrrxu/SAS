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
    X: np.ndarray  # [N, T+1], state at each timestep
    A: np.ndarray  # [N, K+1], ancestor at each timestep
    Y: np.ndarray  # [N, K], observation at each timestep
    
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
        # get observed indices
        self.observed_ind = self.model_interface.observed_ind
        # get observation
        self.outflux = self.model_interface.outflux

        # initialize state
        A = np.zeros((self.N, self.K + 1)).astype(int)
        A[:,0] = np.arange(self.N)
        self.state_init = State( 
            R=self.R,
            W=np.log(np.ones(self.N) / self.N),
            X=np.ones((self.N, self.T+1)) * self._state_init,
            Y=np.zeros((self.N, self.T)),
            A=A
        )
        t_k_map = np.arange(0,self.T+1)[self.observed_ind]
        # if there are no gap in data, do nothing
        post_ind = t_k_map + 1
        if self.K == self.T:
            pre_ind = t_k_map
        # if the first one is observed:
        elif t_k_map[0] == 0:
            pre_ind = t_k_map - 1
            pre_ind[0] = 0
        else:
            pre_ind = t_k_map + 1
            pre_ind = np.insert(pre_ind, 0, 0)
            pre_ind = pre_ind[:-1]
        self.pre_ind = pre_ind
        self.post_ind = post_ind

        # TODO: remove after instant gap is done
        print("pre_ind: ", self.pre_ind)
        print("post_ind: ", self.post_ind)


    def run_sequential_monte_carlo(self) -> None:
        """Run sequential Monte Carlo
        """
        R = self.state_init.R
        W = self.state_init.W
        X = self.state_init.X
        A = self.state_init.A
        Y = self.state_init.Y

        xk = X[A[:,0], 0:1]

        for k in range(self.K):
            start_ind = self.pre_ind[k]
            end_ind = self.post_ind[k]


            # TODO: may need more more work on this cuz currently only one flux
            xkp1 = self.model_interface.transition_model(Xtm1=xk,
                                                         Rt=R[A[:,k], start_ind:end_ind])

            Y[:,start_ind:end_ind] = self.model_interface.observation_model(Xk=xkp1)
            wkp1 = W + np.log(
                #TODO: change it to only give one
                self.model_interface.observation_model_likelihood(
                                                        yhk=Y[:,end_ind-1],
                                                        yk=self.outflux[end_ind-1]
                                                        )
                            )
            W = wkp1
            # This p1 takes account for initial condition
            X[:,start_ind+1:end_ind+1] = xkp1
            A[:,k+1] = _inverse_pmf(A[:,k],W, num = self.N)
            # draw new state samples and associated weights based on last ancestor 
            xk = X[A[:,k+1], start_ind+1:end_ind+1]


        self.state = State(X=X, A=A, W=W, R=R, Y=Y)  
        
        # # generate new set of R
        # self.model_interface.input_model()
        # self.R = self.model_interface.R    
        return 

    def run_particle_MCMC(self) -> None:
        """Run particle MCMC
        """
        R = self.state_init.R
        W = self.state_init.W
        X = self.state_init.X
        A = self.state_init.A
        Y = self.state_init.Y

        # sample an ancestral path based on final weight
        B = self._find_traj(A, W)
        # reinitialize weight
        W = np.log(np.ones(self.N)/self.N)
        # TODO: this is the place to pass a small interval to the transition model
        for k in range(self.K):
            
            start_ind = self.pre_ind[k]
            end_ind = self.post_ind[k]

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
            Y[:,k] = self.model_interface.observation_model(Xk=X[:,k+1])
            wkp1 = W + np.log(
                self.model_interface.observation_model_likelihood(
                                                        yhk=Y[:,k],
                                                        yk=self.outflux[k]
                                                        )
                            )
            W = wkp1#/wkp1.sum()
        
        self.state = State(X=X, A=A, W=W, R=R, Y=Y)  
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
        traj_X = np.zeros(self.T+1)
        t_k_map = self.observed_ind + 1
        t_k_map = np.append(0, t_k_map)
        traj_X[0] = X[B[0],0]
        for i in range(1, self.K+1):
            traj_X[t_k_map[i-1]:t_k_map[i]] = X[B[i],t_k_map[i-1]:t_k_map[i]]
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
        t_k_map = self.observed_ind
        for i in range(self.K-1):
            traj_Y[t_k_map[i]:t_k_map[i+1]] = Y[B[i+1],t_k_map[i]:t_k_map[i+1]]
        traj_Y[t_k_map[-1]:] = Y[B[-1],t_k_map[-1]:]

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
        t_k_map = self.observed_ind
        for i in range(self.K-1):
            traj_R[t_k_map[i]:t_k_map[i+1]] = R[B[i+1],t_k_map[i]:t_k_map[i+1]]
        traj_R[t_k_map[-1]:] = R[B[-1],t_k_map[-1]:]
        return  traj_R

# %%
