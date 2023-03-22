# %%
import pandas as pd
import numpy as np
from typing import List
import scipy.stats as ss
import tqdm
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
        for key in config.keys():
            self.config[key] = config[key]
        else:
            self.config = config
        
        # set influx and outflux here
        # TODO: flexible way to insert multiple influx and outflux
        self.influx = self.df[config['influx']]
        self.outflux = self.df[config['outflux']]

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
        self._num_theta_to_estimate = self._theta_init['to_estimate'].keys()
        
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
                self.update_model[key] = ss.norm(loc = 0, scale = current_theta['update_params'][1])
            else:
                raise ValueError("This search distribution is not implemented yet")
        



    def run_pGS_SAEM(self, num_particles:int,num_params:int, len_MCMC: int, theta_init:dict, q_step:list or float = 0.75):
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
                                'sig_w':{"prior_dis": "uniform", "prior_params":[0.00005,0.0005], 
                                     "update_dis": "normal", "update_params":[0.00005]}},
                'not_to_estimate': {'sig_v': 0.254*1./24/60*15}
            }
        # process theta
        self._theta_init = theta_init
        self._process_theta(theta_init)

        # initialize record storage
        self.theta_record = np.zeros((self.L+1, self._num_theta_to_estimate))
        self.input_record = np.zeros(self.L+1) # assume one input for now
        
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
            q_step = [q_step]*self.L+1 
        Qh = q_step[0] * np.max(WW[:,:],axis = 1)

        for ll in tqdm(range(self.L)):
            # generate random variables now
            theta_record_ll = self.theta_record[ll,:]        
            theta_new = theta_record_ll + self.update_model.rvs(self.D)
            # ======update=========
            # for each parameter
            for p in range(self._num_theta_to_estimate):
                # for each particle
                for d in range(self.D):
                    XX[d,:,:], AA[d,:,:], WW[d,:], RR[d,:,:] = self.run_pMCMC(theta_new, p, XX[d,:,:] , WW[d,:], AA[d,:,:],RR[d,:,:])

                Qh = (1-q_step[ll+1])*Qh + q_step[ll+1] * np.max(WW[:,ll+1,:],axis = 1)
                self.theta_record[ll+1,p] = theta_new[np.argmax(Qh)]
                self.input_record[ll+1,p] = _find_MLE()
        return



# %%
import numpy as np

def gibbs_sampler(p, num_iterations, initial_values, conditional_sampler_functions):


    # Initialize the parameter samples matrix
    parameter_samples = np.zeros((num_iterations, p))

    # Set the initial values for the parameters
    current_values = initial_values

    # Run the Gibbs sampler for num_iterations
    for i in range(num_iterations):
        # Sample from the full conditional distribution of each parameter in turn
        for j in range(p):
            current_values[j] = conditional_sampler_functions[j](current_values)

        # Add the current parameter values to the parameter samples matrix
        parameter_samples[i, :] = current_values

    return parameter_samples