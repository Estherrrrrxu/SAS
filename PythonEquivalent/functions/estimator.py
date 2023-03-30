# %%
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
import scipy.stats as ss
from tqdm import tqdm
from functions.link import ModelLink
from dataclasses import dataclass
from abc import ABC


# %%
@dataclass
class State:
    """ Class for keep track of state at each timestep"""
    R: np.ndarray  # [T, N]
    W: np.ndarray  # [N]
    X: np.ndarray  # [T+1, N]
    A: np.ndarray  # [T+1, N]


class InputModel(ABC):
    """General input model

        for given \theta, r_t = func_\theta(u_t)
    """
    def __init__(self, theta: float):
        """Set theta

        Args:
            theta (np.ndarray): parameter of the model  
        """
        self.theta = theta
    
    def input(self, ut: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            u (np.ndarray): forcing
            t (int): transition timestep

        Returns:
            np.ndarray: r_t = func_\theta(u_t)
        """
        ...

class TransitionModel(ABC):
    """General transition model

    for given \theta, x_t = f_\theta(x_{t-1}, u_t, r_t)
    """
    def __init__(self, theta: float):
        """Set theta

        Args:
            theta (np.ndarray): parameter of the model  
        """
        self.theta = theta
    
    def transition(self, x: np.ndarray, u: np.ndarray, r: np.ndarray, t: int) -> np.ndarray:
        """_summary_

        Args:
            x (np.ndarray): state variable
            u (np.ndarray): forcing
            r (np.ndarray): input uncertainty
            t (int): transition timestep

        Returns:
            np.ndarray: x_t = f_\theta(x_{t-1}, u_t, r_t)
        """
        ...


class ObservationModel(ABC):
    """General observation model

    for given \theta, z_k = g_\theta(x_k, u_k, v_k)
    """
    def __init__(self, theta: np.ndarray = None):
        """Set theta

        Args:
            theta (np.ndarray): parameter of the model  
        """
        self.theta = theta

    def observation(self, x: np.ndarray, u: np.ndarray, v: np.ndarray, k: int) -> np.ndarray:
        """Run model

        Args:
            x (np.ndarray): state variable
            u (np.ndarray): forcing
            v (np.ndarray): associated uncertainty
            k (int): observed timestep
            
        Returns:
            np.ndarray: z_k = g_\theta(x_k, u_k, v_k)
        """
        ...


class ProposalModel(ABC):
    """A class used to construct proposal model
        
    Attributes
    ----------        
    transition_model (TransitionModel):
    observation_model (ObservationModel):
    theta (np.ndarray): parameter of the model

    Methods
    -------
    f_theta()
        Transition model
    g_theta()   
        Observation model
    """
    def __init__(
            self, 
            transition_model: TransitionModel,
            observation_model: ObservationModel,
            theta: np.ndarray
            ) -> None:
        """Initialize the model

        Args:
            transition_model (TransitionModel): 
            observation_model (ObservationModel): 
            theta (np.ndarray, optional): parameter. Defaults to None.
        """
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.theta = theta
    def f_theta(self, xtm1: np.ndarray, ut: np.ndarray) -> np.ndarray:
        ...

    def g_theta(self, yht: np.ndarray, yt: np.ndarray) -> np.ndarray:
        ...

class InputProcess(ABC):
    def ___init__(self, df: pd.DataFrame, config: Optional[dict[str, Any]] = None):
        self.df = df
        self.config = config
    def _process_config(self) -> None:
        ...
    def _process_data(self) -> None:
        ...
    def _process_theta(self, theta_init) -> None:
        ...

# %%
def _inverse_pmf(x: np.ndarray,ln_pmf: np.ndarray, num: int) -> np.ndarray:
    """Sample x based on its ln(pmf) using discrete inverse sampling method

    Args:
        x (np.ndarray): The specific values of x
        pmf (np.ndarray): The weight (ln(pmf)) associated with x
        num (int): The total number of samples to generate

    Returns:
        np.ndarray: index of x that are been sampled according to its ln(pmf)
    """
    ind = np.argsort(x) # sort x according to its magnitude
    pmf = np.exp(ln_pmf) # convert ln(pmf) to pmf
    pmf /= pmf.sum()
    pmf = pmf[ind] # sort pdf accordingly
    u = np.random.uniform(size = num)
    ind_sample = np.searchsorted(pmf.cumsum(), u)
    return ind[ind_sample]

# %%
class SSModel(ABC):
    def __init__(
        self,
        num_input_scenarios: int,
        proposal_model: ProposalModel(TransitionModel, ObservationModel,theta=None),
        input_model: InputModel,
        input_process: InputProcess
    ):
        
        self.N = num_input_scenarios
        self.input_model = input_model
        self.proposal_model = proposal_model
        self.input_process = input_process

    def run_sequential_monte_carlo(self,
        influx: pd.Series,
        outflux: pd.Series,
        input_theta: np.ndarray,
        obs_theta: np.ndarray,
        K: int,
    ) -> State:
        """Run sequential Monte Carlo
        
        Args:
            influx (pd.Series): inflow time series
            outflux (pd.Series): outflow time series
            theta (Tuple[float, float]): parameter of the model
            K (int): number of observations

        Returns:
            State: state at each timestep
        """
        self.K = K
        self.T = len(influx)
        self.influx = influx
        self.outflux = outflux

        self.input_model.theta = input_theta
        self.proposal_model.theta = obs_theta

        # state storage
        # TODO: make this more flexible, maybe add 'state_init' class???
        X = np.zeros((self.N, self.T + 1))
        X[:, 0] = np.ones(self.N) * outflux[0]

        # ancestor storage
        A = np.zeros((self.N, self.K + 1)).astype(int)
        A[:, 0] = np.arange(self.N)

        # other states
        R = np.zeros((self.N, self.T)) # input uncertainty
        W = np.log(np.ones(self.N) / self.N) # last weight, initial weight on particles are all equal

        # TODO: work on diff k and T later
        for k in range(self.K):
            # draw new state samples and associated weights based on last ancestor
            xk = X[A[:,k],k]
            wk = W
            # compute uncertainties 
            R[:,k] = self.input_model.input(self.influx[k])
            # updates
            xkp1 = self.proposal_model.f_theta(xk,R[:,k])
            wkp1 = wk + np.log(self.proposal_model.g_theta(xkp1, self.outflux[k]))
            W = wkp1
            X[:,k+1] = xkp1
            A[:,k+1] = _inverse_pmf(A[:,k],W, num = self.N)
        
        state = State(X, A, W, R)        
        return state
    
    def run_particle_MCMC(
            self, 
            state: State, 
            theta: np.ndarray
            ) -> State:
        """Run particle MCMC
        
        Args:
            state (State): state at each timestep
            theta (np.ndarray): parameter of the model
        
        Returns:
            State: state at each timestep
        """

        # generate random variables now
        X = state.X
        W = state.W
        A = state.A
        R = state.R
        # sample an ancestral path based on final weight
        B = self._find_traj(A,W)
        # state estimation-------------------------------
        W = np.log(np.ones(self.N)/self.N) # initial weight on particles are all equal
        for k in range(self.K):
            rr = self.input_model.input(self.influx[k])
            xkp1 = self.proposal_model.f_theta(X[:,k],R[:,k])
            # Update 
            x_prime = X[B[k+1],k+1]
            # TODO: design a W_tilde module
            W_tilde = W + np.log(ss.norm(x_prime,0.000005).pdf(xkp1))

            A[B[k+1],k+1] = _inverse_pmf(xkp1 - x_prime,W_tilde, num = 1)
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
            wkp1 = W + np.log(self.g_theta(xkp1, self.outflux[k]))
            W = wkp1#/wkp1.sum()
        
        state = State(X, A, W, R)        
        return state



    def run_particle_Gibbs_SAEM(
            self,
            num_parameter_samples:int,
            len_MCMC: int,
            num_theta_to_estimate: dict = None,
            theta_to_estimate: dict = None,
            q_step: np.ndarray or float = 0.75
            ) -> None:
        """ Run particle Gibbs with Ancestor Resampling (pGS) and Stochastic Approximation of the EM algorithm (SAEM)

        Parameters (only showing inputs)
        ----------
        df : pd.DataFrame
            A dataframe contains all essential information as the input 
        config : dict
            A dictionary contains information about parameters to estimate and not to be estimated
    
        """
        # specifications
        self.L = len_MCMC
        self.D = num_parameter_samples
        self._num_theta_to_estimate = num_theta_to_estimate
        self._theta_to_estimate = theta_to_estimate


        # initialize a bunch of temp storage 
        AA = np.zeros((self.D,self.N,self.K+1)).astype(int) 
        WW = np.zeros((self.D,self.N))
        XX = np.zeros((self.D,self.N,self.K+1))
        RR = np.zeros((self.D,self.N,self.K))
        
        # initialize record storage
        self.theta_record = np.zeros((self.L+1, self._num_theta_to_estimate))
        # TODO: assume one input for now
        self.input_record = np.zeros((self.L+1, self._num_theta_to_estimate ,self.T))
        
        # initialize theta
        theta = np.zeros((self.D, self._num_theta_to_estimate))
        for i, key in enumerate(self._theta_to_estimate):
            temp_model = self.prior_model[key]
            theta[:,i] = temp_model.rvs(self.D)

        # run sMC algo first

        for d in range(self.D):
            state = self.run_sequential_monte_carlo(theta[d,:])
            XX[d,:,:],AA[d,:,:],WW[d,:],RR[d,:,:] = state.X, state.A, state.W, state.R

        # temp memory term
        if isinstance(q_step,float):
            q_step = [q_step]*(self.L+1)
        Qh = q_step[0] * np.max(WW[:,:],axis = 1)

        ind_best_param = np.argmax(Qh)
        self.theta_record[0,:] = theta[ind_best_param,:]
 
        for ll in tqdm(range(self.L)):
            # for each parameter
            for p,key in enumerate(self._theta_to_estimate):   
                theta_new = self._process_theta_at_p(p,ll,key)
                # for each particle
                for d in range(self.D):
                    state = State(X = XX[d,:,:] , W = WW[d,:], A = AA[d,:,:],R = RR[d,:,:])
                    state_new = self.run_particle_MCMC(theta = theta_new[d,:], state = state)
                    XX[d,:,:], AA[d,:,:], WW[d,:], RR[d,:,:] = state_new.X, state_new.A, state_new.W, state_new.R

                Qh = (1-q_step[ll+1])*Qh + q_step[ll+1] * np.max(WW[:,:],axis = 1)
                ind_best_param = np.argmax(Qh)
                self.theta_record[ll+1,p] = theta_new[ind_best_param,p]

                B = self._find_traj(AA[ind_best_param,:,:], WW[ind_best_param,:])
                traj_R = self._get_R_traj(RR[ind_best_param,:,:], B)
                self.input_record[ll+1,p,:] = traj_R
        return

    def _process_theta_at_p(self,p,ll,key):
        theta_temp = np.ones((self.D,self._num_theta_to_estimate)) * self.theta_record[ll,:]
        theta_temp[:,:p] = self.theta_record[ll+1,p]
        theta_temp[:,p] += self.update_model[key].rvs(self.D)
        return theta_temp

    def _find_traj(self, A: np.ndarray,W: np.ndarray) -> np.ndarray:
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
        for i in range(self.K+1):
            traj_R[i] = R[B[i+1],i]
        return  traj_R
    

