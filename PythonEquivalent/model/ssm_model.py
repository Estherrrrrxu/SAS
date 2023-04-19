# %%
from typing import Optional
from input_generator import InputGenerator
from model_interface import ModelInterface
from utils_chain import Chain
import numpy as np
from multiprocessing import Pool
# %%
class SSModel:
    def __init__(
        self,
        model_interface: ModelInterface,
        num_parameter_samples: Optional[int] = 15,
        len_parameter_MCMC: Optional[int] = 20,
        learning_step: np.ndarray or float = 0.75
    ) -> None:      
        """Initialize the SSM model taking essential inputs

        Args:
            model_interface (ModelInterface): *initialized* model interface
            num_parameter_samples (int): number of parameter samples, D
            len_parameter_MCMC (int): length of parameter MCMC chain, L
            learning_step (np.ndarray or float): learning step size for SAEM

        """
        # set up input metrics
        self.D = num_parameter_samples
        self.L = len_parameter_MCMC
        self.learning_step = learning_step
        
        # get info from model_interface
        self._num_theta_to_estimate = model_interface._num_theta_to_estimate
        self._theta_to_estimate = model_interface._theta_to_estimate
        self.T = model_interface.T
        self.N = model_interface.N
        self.K = model_interface.K
        # pass model structure for each chain
        self.models_for_each_chain = [model_interface for d in range(self.D)]
        # store necessary models
        self.prior_model = model_interface.prior_model
        self.search_model = model_interface.search_model
        self.R = model_interface.input_model()
     
        # initialize record for parameter and input
        self.theta_record = np.zeros(
            (self.L+1, self._num_theta_to_estimate)
            )
        # TODO: assume one input for now
        self.input_record = np.zeros(
            (self.L+1, self._num_theta_to_estimate, self.T)
            )
        # TODO: write output and state record
        


    def _sample_theta_from_prior(self) -> np.ndarray:
        """Sample theta from their prior distribution

        Returns:
            np.ndarray: sampled new theta candidates
        """
        theta_new = np.zeros(self._num_theta_to_estimate)
        for i, key in enumerate(self._theta_to_estimate):
             theta_new[i] = self.prior_model[key].rvs()
        return theta_new

    def run_particle_Gibbs_AR_SAEM(self) -> None:
        """ Run particle Gibbs with Ancestor Resampling (AR) and Stochastic Approximation of the EM algorithm (SAEM)    
        """

        # initilize D chains
        def initialize_chains(model):
            theta_new = self._sample_theta_from_prior()
            chain = Chain(model, theta_new, self.R)

            self._update_model(theta_new)
            return
        with Pool(processes=self.D) as pool:
            theta_new = pool.map(
                initialize_chains, self.model_links
            )


# input will be passed through input model
        
        # initialize theta
        #theta_new = np.zeros((self.D, self._num_theta_to_estimate))
        #for i, key in enumerate(self._theta_to_estimate):
            #temp_model = self.model_links[0].prior_model[key]
            #theta_new[:,i] = temp_model.rvs(self.D)
        # run sMC algo first
        #theta_new = np.array([model_link.sample_theta_from_prior() for model_link in self.model_links])


        # initialize a bunch of temp storage 
        AA = np.zeros((self.D,self.N,self.K+1)).astype(int) 
        WW = np.zeros((self.D,self.N))
        XX = np.zeros((self.D,self.N,self.K+1))
        RR = np.zeros((self.D,self.N,self.K))
        



 