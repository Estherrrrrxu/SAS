# %%
from typing import Optional
from tqdm import tqdm
from model.model_interface import ModelInterface
from model.utils_chain import Chain
import numpy as np
import matplotlib.pyplot as plt
# %%
class SSModel:
    def __init__(
        self,
        model_interface: ModelInterface,
        num_parameter_samples: Optional[int] = 15,
        len_parameter_MCMC: Optional[int] = 20,
        fast_convergence_phase_length: Optional[int] = None
    ) -> None:      
        """Initialize the SSM model taking essential inputs

        Args:
            model_interface (ModelInterface): *initialized* model interface
            num_parameter_samples (int): number of parameter samples, D
            len_parameter_MCMC (int): length of parameter MCMC chain, L
            fast_convergence_phase_length (int): length of fast convergence phase

        """
        # set up input metrics
        self.D = num_parameter_samples
        self.L = len_parameter_MCMC
        self.fast_convergence_phase_length = fast_convergence_phase_length
        if self.fast_convergence_phase_length is None:
            self.fast_convergence_phase_length = self.L
        
        self.learning_step = np.ones(self.L+1)
        if self.fast_convergence_phase_length < self.L:
            for i in range(self.fast_convergence_phase_length+1, self.L+1):
                self.learning_step[i] = 1. / (i - self.fast_convergence_phase_length)

        
        # get info from model_interface
        self._num_theta_to_estimate = model_interface._num_theta_to_estimate
        self._theta_to_estimate = model_interface._theta_to_estimate
        self.T = model_interface.T
        self.N = model_interface.N
        self.K = model_interface.K

        # pass model structure for each chain
        self.models_for_each_chain = [model_interface for d in range(self.D)]
        # store necessary models
        self.dist_model = model_interface.dist_model

        # initialize record for parameter and input
        self.theta_record = np.zeros(
            (self.L+1, self._num_theta_to_estimate)
            )
        
        # TODO: assume one input for now
        self.input_record = np.zeros(
            (self.L+1, self.T)
            )

        self.state_record = np.zeros(
            (self.L+1, self.T+1)
            )

        self.output_record = np.zeros(
            (self.L+1, self.T)
            )



    def _sample_theta(self) -> np.ndarray:
        """Sample theta from given distribution

        Returns:
            np.ndarray: sampled new theta candidates
        """
        theta_new = np.zeros(self._num_theta_to_estimate)
        for i, key in enumerate(self._theta_to_estimate):
             theta_new[i] = self.dist_model[key].rvs()
        return theta_new
    
    
    def run_particle_Gibbs_AS_SAEM(self) -> None:
        """ Run particle Gibbs with Ancestor Sampling (AS) and Stochastic Approximation of the EM algorithm (SAEM)    
        """
        # initialize likelihood, theta storage space, and chains
        WW = np.zeros((self.D, self.N))
        theta_new = np.zeros((self.D, self._num_theta_to_estimate)) 
        chains = [Chain(model_interface=self.models_for_each_chain[d]) for d in range(self.D)]

        # draw D theta candidates for each chain, and run sMC
        for d in range(self.D):
            theta_new[d,:] = self._sample_theta()   
            chains[d].model_interface.update_model(theta_new[d,:])      
            chains[d].run_sequential_monte_carlo()
            WW[d,:] = chains[d].state.W

        # find optimal parameters
        Qh = self.learning_step[0] * np.max(WW[:,:],axis = 1)
        ind_best_param = np.argmax(Qh)
        self.theta_record[0,:] = theta_new[ind_best_param, :]

        best_model = chains[ind_best_param]
        B = best_model._find_traj(best_model.state.A, best_model.state.W)
        self.input_record[0,:] = best_model._get_R_traj(best_model.state.R, B)
        self.state_record[0,:] = best_model._get_X_traj(best_model.state.X, B)
        self.output_record[0,:] = best_model._get_Y_traj(best_model.state.Y, B)

        for l in tqdm(range(self.L)):
            # for each theta
            for p, key in enumerate(self._theta_to_estimate):
                theta_new = self._update_theta_at_p(
                        p=p,
                        l=l,
                        key=key
                    )

                # update chains
                for d in range(self.D):
                    # initialize a chain using this theta
                    chains[d].model_interface.update_model(theta_new[d,:])
                    chains[d].run_particle_MCMC()   
                    WW[d,:] = chains[d].state.W

                # find optimal parameters
                # Qh = self.learning_step[l+1] * (np.max(WW[:,:],axis = 1) + np.log(self.dist_model[key].pdf(theta_new[:,p]))) + (1 - self.learning_step[l+1]) * Qh
                Qh = self.learning_step[l+1] * np.max(WW[:,:],axis = 1) + (1 - self.learning_step[l+1]) * Qh
                ind_best_param = np.argmax(Qh)
                self.theta_record[l+1,p] = theta_new[ind_best_param, p]   
                
                print('=============================================')
                print('Qh: ', Qh)
                print('corresponding weights: ', np.max(WW[:,:],axis = 1))
                print('best index: ', ind_best_param)
                print('theta_new: ', theta_new[ind_best_param, :])
                print('=============================================')
                
                
                
                
                best_model = chains[ind_best_param]  


                B = best_model._find_traj(best_model.state.A, best_model.state.W)
                self.input_record[l+1,:] = best_model._get_R_traj(best_model.state.R, B)
                self.state_record[l+1,:] = best_model._get_X_traj(best_model.state.X, B)
                self.output_record[l+1,:] = best_model._get_Y_traj(best_model.state.Y, B)
        self.dist_model = best_model.model_interface.update_parameter_distribution()


    def _update_theta_at_p(
            self,
            p: int,
            l: int,
            key: str,
        ) -> np.ndarray:
        """Generate random p-th theta

        Args:
            p (int): index of theta to update
            l (int): index of current MCMC chain
            key (str): key of theta to update
        
        Returns:
            np.ndarray: new theta candidates
        """
        theta_temp = np.ones((self.D,self._num_theta_to_estimate)) * self.theta_record[l,:]
        theta_temp[:,:p] = self.theta_record[l+1,:p]
        # add random noise to the p-th theta
        temp = self.dist_model[key].rvs(self.D)
        theta_temp[:,p] = temp
        return theta_temp
                
 
# %%
