# %%
import pandas as pd
import numpy as np
from typing import List
import scipy.stats as ss
from tqdm import tqdm
from functions.link import input_model, transition_model, observation_model
# %%
# ==========================
# utility functions
# ==========================
# discrete inverse transform sampling
def inverse_pmf(ln_pmf,x,num):
    '''
        give x ~ ln(pdf)
    return: index of x that are been sampled according to pdf
    '''
    pmf = np.exp(ln_pmf)
    pmf = pmf/pmf.sum()
    ind = np.argsort(x) # sort x according to its magnitude
    pmf = pmf[ind] # sort pdf accordingly
    cdf = pmf.cumsum()
    u = np.random.uniform(size = num)
    a = np.zeros(num)
    for i in range(num):
        if u[i] > cdf[-1]:
            a[i] = -1
        elif u[i] < cdf[0]: # this part can be removed
            a[i] = 0
        else:
            # TODO: any more efficient method?
            for j in range(1,len(cdf)):
                if (u[i] <= cdf[j]) and (u[i] > cdf[j-1]):
                    a[i] = j
                    break
    return ind[a.astype(int)]
# backtracking
def find_MLE(X,A,W,R):
    length = A.shape[1]
    B = np.zeros(length).astype(int)
    B[-1] = inverse_pmf(W,A[:,-1], num = 1)
    for i in reversed(range(1,length)):
        B[i-1] =  A[:,i][B[i]]
    MLE = np.zeros(length)
    for i in range(length):
        MLE[i] = X[B[i],i]
    MLE_R = np.zeros(length-1)
    for i in range(length-1):
        MLE_R[i] = R[B[i+1],i]
    return MLE, MLE_R

# %%
# ==========================
# general model class
# ==========================
class SSModel:
    def __init__(self, data_df:pd.DataFrame, config:dict = None):
        '''
            inputs:
                    - data_df: the input time series
                    - config: configuration
                            - if none given, use default_configs
            variables:
                    - df: dataframe that contains all data
                    - 
        '''
        self.df = data_df
        self._default_configs = {
            'dt': 1./24/60*15,
            'influx': 'J_obs',
            'outflux': 'Q_obs'
        }
               
        self.config = self._default_configs
        # TODO: find a better way to write this
        if config is not None:
            for key in config.keys():
                self.config[key] = config[key]
            else:
                self.config = config
        
        # set influx and outflux here
        # TODO: flexible way to insert multiple influx and outflux
        self.influx = self.df[self.config['influx']]
        self.outflux = self.df[self.config['outflux']]

        # TODO: observation currently made at every timestep, adjust it according to interval later
        self.K = len(self.influx)
        # adjust delta_t regarding to time interval
        self.dt = self.config['dt']
    
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
        """

        :param num_particles: number of input scenarios represented by particles
        :param num_params: number of samples of parameters to estimate
        :param len_MCMC: the length of Markov chain
        :param theta_int: a dictionary to initialize and process theta


        :return: 
                    
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
                                     "update_dis": "normal", "update_params":[0.05]},
                                'output_uncertainty':{"prior_dis": "uniform", "prior_params":[0.00005,0.0005], 
                                     "update_dis": "normal", "update_params":[0.00005]}},
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
            R[:,kk] = input_model(self.influx[kk],self._theta_init, self.N)
            # updates
            xkp1 = self.f_theta(xk,theta_to_estimate,R[:,kk])
            wkp1 = wk + np.log(self.g_theta(xkp1, theta_to_estimate, self.outflux[kk]))
            W = wkp1
            X[:,kk+1] = xkp1
            aa = inverse_pmf(W,A[:,kk], num = self.N)
            A[:,kk+1] = aa        
        return X,A,W,R
    
    def f_theta(self,xht:List[float],theta_to_estimate:float, rtp1:float):
        theta_val = theta_to_estimate[0] # specify the variable
        xhtp1 = transition_model(xht, theta_val, self.config['dt'], rtp1)
        return xhtp1

    def g_theta(self, xht:List[float],theta_to_estimate:dict,xt:List[float]):
        theta_val = theta_to_estimate[1]
        likelihood = observation_model(xht,theta_val,xt)
        return likelihood

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
        B[-1] = inverse_pmf(W,A[:,-1], num = 1)
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
            A[B[kk+1],kk+1] = inverse_pmf(W_tilde,xkp1 - x_prime, num = 1)
            # now update everything in new state
            notB = np.arange(0,self.N)!=B[kk+1]
            A[:,kk+1][notB] = inverse_pmf(W,X[:,kk], num = self.N-1)
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
# %%
