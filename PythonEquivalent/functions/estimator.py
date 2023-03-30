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
            theta: np.ndarray = None
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


# %%
class SSModel(ABC):
    def __init__(
        self,
        num_input_scenarios: int,
        proposal_model: ProposalModel(TransitionModel, ObservationModel,theta=None),
        input_model: InputModel
    ):
        
        self.N = num_input_scenarios
        self.input_model = input_model
        self.proposal_model = proposal_model


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
            aa = _inverse_pmf(A[:,k],W, num = self.N)
            A[:,k+1] = aa
        
        state = State(X, A, W, R)        
        return state
    
    def run_pMCMC(self, theta:List[float], X: List[float], W: List[float], A: List[float],R: List[float]):
        '''
        pMCMC inside loop, run this entire function as many as possible
        update w/ Ancestor sampling
            theta           - let it be k, sig_v, sig_w for now
            For reference trajectory:
            qh              - estiamted state in particles
            P               - weight associated with each particles
            A               - ancestor lineage
            p               - the p-th parameter to estimate
            
        '''
        # generate random variables now
        X = X.copy()
        W = W.copy()
        A = A.copy()
        R = R.copy()
        # sample an ancestral path based on final weight

        B = np.zeros(self.K+1).astype(int)
        B[-1] = _inverse_pmf(A[:,-1],W, num = 1)
        for i in reversed(range(1,self.K+1)):
            B[i-1] =  A[:,i][B[i]]
        # state estimation-------------------------------
        W = np.log(np.ones(self.N)/self.N) # initial weight on particles are all equal
        for kk in range(self.K):
            rr = input_model(self.influx[kk],self._theta_init, self.N)
            xkp1 = self.f_theta(X[:,kk],theta,R[:,kk])
            # Look into this
            x_prime = X[B[kk+1],kk+1]
            W_tilde = W + np.log(ss.norm(x_prime,0.000005).pdf(xkp1))
            # ^^^^^^^^^^^^^^
            A[B[kk+1],kk+1] = _inverse_pmf(xkp1 - x_prime,W_tilde, num = 1)
            # now update everything in new state
            notB = np.arange(0,self.N)!=B[kk+1]
            A[:,kk+1][notB] = _inverse_pmf(X[:,kk],W, num = self.N-1)
            xkp1[notB] = xkp1[A[:,kk+1][notB]]
            rr[notB] = rr[A[:,kk+1][notB]]
            xkp1[~notB] = xkp1[A[B[kk+1],kk+1]]
            rr[~notB] = R[:,kk][A[B[kk+1],kk+1]]
            X[:,kk+1] = xkp1   
            R[:,kk] = rr       
            W[notB] = W[A[:,kk+1][notB]]
            W[~notB] = W[A[B[kk+1],kk+1]]
            wkp1 = W + np.log(self.g_theta(xkp1, theta, self.outflux[kk]))
            W = wkp1#/wkp1.sum()
        return X, A, W, R
        


    def run_pGS_SAEM(self, num_particles:int,num_params:int, len_MCMC: int, theta_init:dict = None, q_step:list or float = 0.75):
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
        self.D = num_params
        self.N = num_particles

        # initialize a bunch of temp storage 
        AA = np.zeros((self.D,self.N,self.K+1)).astype(int) 
        WW = np.zeros((self.D,self.N))
        XX = np.zeros((self.D,self.N,self.K+1))
        RR = np.zeros((self.D,self.N,self.K))
        
        if theta_init == None:
            # set default theta
            theta_init = {
                'to_estimate': {'k':{"prior_dis": "normal", "prior_params":[1.2,0.3], 
                                     "update_dis": "normal", "update_params":[0.05],
                                     'log':False},
                                'output_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.0005], 
                                     "update_dis": "normal", "update_params":[0.00005],
                                     'log': True}
                                },
                'not_to_estimate': {'input_uncertainty': 0.254*1./24/60*15}
            }
        # process theta
        self._theta_init = theta_init
        self._process_theta()

        # initialize record storage
        self.theta_record = np.zeros((self.L+1, self._num_theta_to_estimate))
        # TODO: change self.K to self.T in future
        self.input_record = np.zeros((self.L+1, self._num_theta_to_estimate ,self.K)) # assume one input for now
        
        # initialize theta
        theta = np.zeros((self.D, self._num_theta_to_estimate))
        for i, key in enumerate(self._theta_to_estimate):
            temp_model = self.prior_model[key]
            theta[:,i] = temp_model.rvs(self.D)

        # run sMC algo first
        for d in range(self.D):
            XX[d,:,:],AA[d,:,:],WW[d,:],RR[d,:,:] = self.run_sMC(theta[d,:])

        # temp memory term
        if isinstance(q_step,float):
            q_step = [q_step]*(self.L+1)
        Qh = q_step[0] * np.max(WW[:,:],axis = 1)

        ind_best_param = np.argmax(Qh)
        self.theta_record[0,:] = theta[ind_best_param,:]
 # TODO: getter and setter for theta
        for ll in tqdm(range(self.L)):
            # for each parameter
            for p,key in enumerate(self._theta_to_estimate):   
                theta_new = self._process_theta_at_p(p,ll,key)
                # for each particle
                for d in range(self.D):
                    XX[d,:,:], AA[d,:,:], WW[d,:], RR[d,:,:] = self.run_pMCMC(theta_new[d,:], XX[d,:,:] , WW[d,:], AA[d,:,:],RR[d,:,:])

                Qh = (1-q_step[ll+1])*Qh + q_step[ll+1] * np.max(WW[:,:],axis = 1)
                ind_best_param = np.argmax(Qh)
                self.theta_record[ll+1,p] = theta_new[ind_best_param,p]
                _, MLE_R = find_MLE(XX[ind_best_param,:,:], AA[ind_best_param,:,:], WW[ind_best_param,:], RR[ind_best_param,:,:])
                self.input_record[ll+1,p,:] = MLE_R
        return


    








    def _find_traj(self, A: List[str],W: List[str]):
        """Find particle trajectory based on final weight

        Parameters
        ----------
        A : List[str]
            Record of particles move to the next time step
        W : List[str]
            Weight associated with each particle at final timestep

        Returns
        -------
        List[int]
            Index of particle through time
        """

        self.B = np.zeros(self.K+1).astype(int)
        self.B[-1] = _inverse_pmf(A[:,-1],W, num = 1)
        for i in reversed(range(1,self.K+1)):
            self.B[i-1] =  A[:,i][self.B[i]]
        return self.B

    def _get_X_traj(self, X: List[str]) -> List[str]:
        """Get X trajectory based on sampled particle trajectory

        Parameters
        ----------
        X : List[str]
            State variable

        Returns
        -------
        List[int]
            Trajectory of X that is sampled at final timestep
        """
        traj_X = np.zeros(self.K+1)
        for i in range(self.K+1):
            traj_X[i] = X[self.B[i],i]
        return traj_X
    
    def _get_R_traj(self, R: List[str]) -> List[str]:
        """Get R trajectory based on sampled particle trajectory

        Parameters
        ----------
        R : List[str]
            Input uncertainty matrix    

        Returns
        -------
        List[int]
            Trajectory of R that is sampled at final timestep
        """
        traj_R = np.zeros(self.K)
        for i in range(self.K+1):
            traj_R[i] = R[self.B[i+1],i]
        return  traj_R
    








# %%
def _inverse_pmf(x: List[float],ln_pmf: List[float], num: int) -> List[int]:
    """Sample x based on its ln(pmf) using discrete inverse sampling method

    Parameters
    ----------
    x : List[str]
        The specific values of x
    pmf : List[str]
        The weight (ln(pmf)) associated with x
    num : int
        The total number of samples to generate

    Returns
    -------
    List[int]
        index of x that are been sampled according to its ln(pmf)
    """
    ind = np.argsort(x) # sort x according to its magnitude
    pmf = np.exp(ln_pmf) # convert ln(pmf) to pmf
    pmf = pmf/pmf.sum()
    pmf = pmf[ind] # sort pdf accordingly
    cmf = pmf.cumsum()
    u = np.random.uniform(size = num)
    ind_sample = np.zeros(num)
    # TODO: any more efficient method? Idea: sort u first and one pass in enumerating cmf
    for i in range(num):
        if u[i] > cmf[-1]:
            ind_sample[i] = -1
        elif u[i] < cmf[0]:
            ind_sample[i] = 0
        else:
            for j in range(1,len(cmf)):
                if (u[i] <= cmf[j]) and (u[i] > cmf[j-1]):
                    ind_sample[i] = j
                    break
    return ind[ind_sample.astype(int)]