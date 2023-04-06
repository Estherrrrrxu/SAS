# %%
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
import scipy.stats as ss
from tqdm import tqdm
from functions.link import ModelLink
from dataclasses import dataclass
from multiprocessing import Pool
from functions.utils import _inverse_pmf
from functools import reduce
# https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
# %%
@dataclass
class State:
    """ Class for keep track of state at each timestep"""
    R: np.ndarray  # [T, N]
    W: np.ndarray  # [N]
    X: np.ndarray  # [T+1, N]
    A: np.ndarray  # [T+1, N]

# %%
class SSModel:
    def __init__(
        self,
        ModelLink,
        num_parameter_samples:int,
        len_parameter_MCMC: int,
        *model_args
    ):
        self.ModelLinkClass = ModelLink
        self.model_args = model_args

        self.L = len_parameter_MCMC
        self.D = num_parameter_samples

        self.model_links = [ModelLink(*model_args) for d in self.D]

        #self.N = model_link.N
        #self.influx = model_link.influx
        #self.outflux = model_link.outflux
        #self.T = model_link.T
        #self.K = model_link.K

    def run_sequential_monte_carlo(
            self,
            model_link
            ) -> State:
        """Run sequential Monte Carlo
        
        Args:
            theta (float): parameter of the model

        Returns:
            State: state at each timestep
        """

        # state storage
        # TODO: make this more flexible, maybe add 'state_init' class???
        X = np.zeros((self.N, self.T + 1))
        X[:, 0] = np.ones(self.N) * self.outflux[0]

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
            R[:,k] = model_link.input_model()
            # updates
            xkp1 = model_link.f_theta(xk)
            wkp1 = wk + np.log(model_link.g_theta())
            W = wkp1
            X[:,k+1] = xkp1
            A[:,k+1] = _inverse_pmf(A[:,k],W, num = self.N)
       
        
        state = State(X = X, A = A, W = W, R = R)      
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
            rr = self.model_link.input_model(self.influx[k])
            xkp1 = self.model_link.f_theta(X[:,k],R[:,k], theta_k=theta[0])
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
            wkp1 = W + np.log(self.model_link.g_theta(xkp1, self.outflux[k], theta_obs=theta[1]))
            W = wkp1#/wkp1.sum()
        
        state = State(X = X, A = A, W = W, R = R)             
        return state



    def run_particle_Gibbs_SAEM(
            self,
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
        self._num_theta_to_estimate = self.model_links[0]._num_theta_to_estimate
        self._theta_to_estimate = self.model_links[0]._theta_to_estimate


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
        #theta_new = np.zeros((self.D, self._num_theta_to_estimate))
        #for i, key in enumerate(self._theta_to_estimate):
            #temp_model = self.model_links[0].prior_model[key]
            #theta_new[:,i] = temp_model.rvs(self.D)
        # run sMC algo first
        #theta_new = np.array([model_link.sample_theta_from_prior() for model_link in self.model_links])

        def initialize_chains(model_link):
            return model_link.sample_theta_from_prior()
        with Pool(processes=self.D) as pool:
            theta_new = pool.map(
                initialize_chains, self.model_links
            )
        # run sMC algo first
        # queue = Queue()
        # p = Process(target=self.run_sequential_monte_carlo, args=(queue, 1))
        # p.start()
        # p.join() # this blocks until the process terminates
        # result = queue.get()

        #with Pool(processes=self.D) as pool:
        #    results = pool.map(
        #        self.run_sequential_monte_carlo, theta_new
        #    )
        with Pool(processes=self.D) as pool:
            results = pool.map(
                self.run_sequential_monte_carlo, self.model_links
            )
        new_states = [t[1] for t in sorted(results, key=lambda t: t[0])]
        XX = np.vstack([s.X for s in new_states], axis=0)
        AA = np.vstack([s.A for s in new_states], axis=0)
        WW = np.vstack([s.W for s in new_states], axis=0)
        RR = np.vstack([s.R for s in new_states], axis=0)
        
        
        # for d in range(self.D):
        #     state = self.run_sequential_monte_carlo(theta_new[d,:])
        #     XX[d,:,:],AA[d,:,:],WW[d,:],RR[d,:,:] = state.X, state.A, state.W, state.R

        # temp memory term
        if isinstance(q_step,float):
            q_step = [q_step]*(self.L+1)
        Qh = q_step[0] * np.max(WW[:,:],axis = 1)

        ind_best_param = np.argmax(Qh)
        self.theta_record[0,:] = theta_new[ind_best_param,:]
 
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
        theta_temp[:,:p] = self.theta_record[ll+1,:p]
        theta_temp[:,p] += self.model_link.update_model[key].rvs(self.D)
        return theta_temp

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
