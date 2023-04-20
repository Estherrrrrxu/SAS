# %%
from typing import Optional
from tqdm import tqdm
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
        if isinstance(self.learning_step,float):
            self.learning_step = [self.learning_step] * (self.L + 1)
        
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
            # put new theta into model
            model.update_model(theta_new)
            # initialize a chain using this theta
            chain = Chain(
                model_interface=model, 
                theta=theta_new, 
                R=self.R
                )
            return chain, theta_new
        
        with Pool(processes=self.D) as pool:
            chain, theta_new = pool.map(
                initialize_chains, 
                self.models_for_each_chain
            )
            chain.run_sequential_monte_carlo()
        
        WW = np.vstack([chain[d].state.W for d in self.D], axis=0)
        # find optimal parameters
        Qh = self.learning_step[0] * np.max(WW[:,:],axis = 1)
        ind_best_param = np.argmax(Qh)
        self.theta_record[0,:] = theta_new[ind_best_param, :]
        

        for l in range(self.L):
            # update chains
            def updating_chains(model):
                # for each theta
                for p, key in enumerate(self._theta_to_estimate):
                    self._update_theta_at_p(
                            p=p,
                            l=l,
                            key=key
                        )

                    # put new theta into model
                    model.update_model(theta_new)
                    # initialize a chain using this theta
                    chain = Chain(
                        model_interface=model, 
                        theta=theta_new, 
                        R=self.R
                        )
                    chain.run_particle_MCMC()   
                    WW = np.vstack([chain[d].state.W for d in self.D], axis=0)
                    # find optimal parameters
                    Qh = self.learning_step[l+1] * np.max(WW[:,:],axis = 1)
                    ind_best_param = np.argmax(Qh)
                    self.theta_record[l,p] = theta_new[ind_best_param, p]   
                    # TODO: update input record    


                return chain, theta_new


            with Pool(processes=self.D) as pool:
                chain, theta_new = pool.map(
                    updating_chains, 
                    self.models_for_each_chain
                )



    def _update_theta_at_p(
            self,
            p,
            l,
            key
        ) -> np.ndarray:
        theta_temp = np.ones((self.D,self._num_theta_to_estimate)) * self.theta_record[l,:]
        theta_temp[:,:p] = self.theta_record[l+1,:p]
        theta_temp[:,p] += self.search_model[key].rvs(self.D)
        return theta_temp
                    



        



 