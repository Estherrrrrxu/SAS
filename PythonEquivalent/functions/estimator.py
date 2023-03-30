# %%
import pandas as pd
import numpy as np
from typing import List, Tuple
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
    
    def input(self, u: np.ndarray, t: int) -> np.ndarray:
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
    def f_theta(self):
        ...

    def g_theta(self):
        ...


# %%
class SSModel(ABC):
    """
    A class used to construct SSModel (State Space Model) algorithm framework

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
    T : int
        Total time length of the time series
    K : int
        Total number of observation made
    dt : float
        Time interval between time steps


    Methods
    -------
    _process_theta()
        Generate the input time series for given period

    transition_model()
        Generate the state to the next set of time

    observation_model()
        Generate observation at given time
    """

    
    # ---------------pGS_SAEM algo----------------
    def _process_theta(self):
        """
            For theta models:
            1. Prior model: 
                normal:     params [mean, std]
                uniform:    params [lower bound, upper bound]
            2. Update model:
                normal:     params [std]
        """
        # find params to update
        self._theta_to_estimate = list(self._theta_init['to_estimate'].keys())
        self._num_theta_to_estimate = len(self._theta_to_estimate )
        
        # save models
        self.prior_model = {}
        self.update_model = {}

        for key in self._theta_to_estimate:
            current_theta = self._theta_init['to_estimate'][key]
            if current_theta['log'] == True:
                # TODO: need to think about how to do log part
                # for prior distribution
                if current_theta['prior_dis'] == 'normal':
                    self.prior_model[key] = ss.norm(loc = np.log(current_theta['prior_params'][0]), scale = np.log(current_theta['prior_params'][1]))
                elif current_theta['prior_dis'] == 'uniform':
                    self.prior_model[key] = ss.uniform(loc = np.log(current_theta['prior_params'][0]),scale = np.log(current_theta['prior_params'][1] - current_theta['prior_params'][0]))
                else:
                    raise ValueError("This prior distribution is not implemented yet")
                
                # for update distributions
                if current_theta['update_dis'] == 'normal':
                    self.update_model[key] = ss.norm(loc = 0, scale = current_theta['update_params'][0])
                else:
                    raise ValueError("This search distribution is not implemented yet")
            else:
                # for prior distribution
                if current_theta['prior_dis'] == 'normal':
                    self.prior_model[key] = ss.norm(loc = current_theta['prior_params'][0], scale = current_theta['prior_params'][1])
                elif current_theta['prior_dis'] == 'uniform':
                    self.prior_model[key] = ss.uniform(loc = current_theta['prior_params'][0],scale = (current_theta['prior_params'][1] - current_theta['prior_params'][0]))
                else:
                    raise ValueError("This prior distribution is not implemented yet")
                
                # for update distributions
                if current_theta['update_dis'] == 'normal':
                    self.update_model[key] = ss.norm(loc = 0, scale = current_theta['update_params'][0])
                else:
                    raise ValueError("This search distribution is not implemented yet")
        
    def _process_theta_at_p(self,p,ll,key):
        theta_temp = np.ones((self.D,self._num_theta_to_estimate)) * self.theta_record[ll,:]
        theta_temp[:,:p] = self.theta_record[ll+1,p]
        theta_temp[:,p] += self.update_model[key].rvs(self.D)
        return theta_temp
        


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


    




class ModelBase:
    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        """
        Parameters (only showing inputs)
        ----------
        df : pd.DataFrame
            A dataframe contains all essential information as the input 
        config : dict
            A dictionary contains information about parameters to estimate and not to be estimated
        """
        # load input dataframe--------------
        self.df = df

        # process configurations------------
        self._default_configs = {
            'dt': 1./24/60*15,
            'influx': 'J_obs',
            'outflux': 'Q_obs',
            'observed_at_each_time_step': True,
            'observed_interval': None,
            'observed_series': None # give a boolean list of observations been made 
        }   
        self.config = self._default_configs
        # replace default config with input configs
        if config is not None:
            for key in config:
                self.config[key] = config[key]
            else:
                self.config = config
        
        # set influx and outflux here--------
        # TODO: flexible way to insert multiple influx and outflux
        self.influx = self.df[self.config['influx']]
        self.outflux = self.df[self.config['outflux']]

        # set observation interval-----------
        self.T = len(self.influx)
        if self.config['observed_at_each_time_step'] == True:
            self.K = self.T
        elif self.config['observed_interval'] is not None:
            self.K = int(self.T/self.config['observed_interval'])
        else:
            self.K = sum(self.config['observed_series']) # total num observation is the number 
            self._is_observed = self.config['observed_series'] # set another variable to indicate observation

        # set delta_t------------------------
        self.dt = self.config['dt']
    

class A(SSModel):
    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        super().__init__(df, config)

    def f_theta(self, xht: np.ndarray, theta_to_estimate: float, rtp1: float):
        theta_val = theta_to_estimate[0]  # specify the variable
        xhtp1 = self.transition_model(xht, theta_val, self.config['dt'], rtp1)
        return xhtp1

    def g_theta(self, xht:List[float],theta_to_estimate:dict,xt:List[float]):
        theta_val = theta_to_estimate[1]
        likelihood = self.observation_model(xht,theta_val,xt)
        return likelihood

    


    def run_sMC(self, theta_to_estimate):
        '''
            :param theta_to_estimate: initial theta to estimate
        '''
        # initialization---------------------------------
        A = np.zeros((self.N,self.K+1)).astype(int) # ancestor storage
        A[:,0] = np.arange(self.N) # initialize the first set of particles
        X = np.zeros((self.N,self.K+1)) # for each particle, for each X
        # TODO: make this more flexible, maybe add 'state_init' in config
        X[:,0] = np.ones(self.N)*self.outflux[0] # for each particle, for each X
        # and we only need to store the last weight
        W = np.log(np.ones(self.N)/self.N) # initial weight on particles are all equal
        R = np.zeros((self.N,self.K)) # store the stochasity from input concentration
        # state estimation-------------------------------
        for kk in range(self.K):
            # draw new state samples and associated weights based on last ancestor
            xk = X[A[:,kk],kk]
            wk = W
            # compute uncertainties TODO: currently theta not being estimated
            R[:,kk] = ModelLink.input_model(self.influx[kk],self._theta_init, self.N)
            # updates
            xkp1 = self.f_theta(xk,theta_to_estimate,R[:,kk])
            wkp1 = wk + np.log(self.g_theta(xkp1, theta_to_estimate, self.outflux[kk]))
            W = wkp1
            X[:,kk+1] = xkp1
            aa = _inverse_pmf(A[:,kk],W, num = self.N)
            A[:,kk+1] = aa        
        return X,A,W,R


class B(ModelBase)
    def __init__(self, df:pd.DataFrame, config:dict = None):
            super.__init__( df:pd.DataFrame, config:dict = None)
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





class C(ModelBase)
    def __init__(self, df:pd.DataFrame, config:dict = None):
            super.__init__( df:pd.DataFrame, config:dict = None)
            self.bs = [B(...)]
        self.a = A(...)
        self.bs = []

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