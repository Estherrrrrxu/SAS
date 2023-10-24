# %%
from typing import Optional
from tqdm import tqdm
from model.model_interface import ModelInterface
from model.utils_chain import Chain
import numpy as np
import multiprocessing as mp
from functions.utils import _inverse_pmf

# %%
def worker_function_sMC(chain_d, theta_new):   
    chain_d.model_interface.update_theta(theta_new)  
    chain_d.run_sequential_monte_carlo()
    
    model_W = chain_d.state.W
    B = chain_d._find_traj(chain_d.state.A, model_W, max=True)
    traj_X = chain_d._get_X_traj(chain_d.state.X, B)
    
    state_W = chain_d.model_interface.transition_model_probability(traj_X)
    WW = state_W + np.log(model_W.max())
    return WW, B
def worker_function_pMCMC(chain_d, theta_new):
    chain_d.model_interface.update_theta(theta_new)
    chain_d.run_particle_MCMC_AS()

    model_W = chain_d.state.W
    B = chain_d._find_traj(chain_d.state.A, model_W, max=True)
    traj_X = chain_d._get_X_traj(chain_d.state.X, B)

    state_W = chain_d.model_interface.transition_model_probability(traj_X)
    WW = state_W + np.log(model_W.max())
    return WW, B

#%%
class SSModel:
    def __init__(
        self,
        model_interface: ModelInterface,
        num_parameter_samples: Optional[int] = 15,
        len_parameter_MCMC: Optional[int] = 20
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
        
        # get info from model_interface
        self._num_theta_to_estimate = model_interface._num_theta_to_estimate
        self._theta_to_estimate = model_interface._theta_to_estimate
        self.T = model_interface.T
        self.N = model_interface.N
        self.K = model_interface.K
        self.is_MAP_MCMC = model_interface.config["use_MAP_MCMC"]
        self.update_theta_dist = model_interface.config["update_theta_dist"]

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
            (self.L+1, self.T)
            )

        self.output_record = np.zeros(
            (self.L+1, self.T)
            )



    def _sample_theta_init(self) -> np.ndarray:
        """Sample theta from given initial prior distribution

        Returns:
            np.ndarray: sampled new theta candidates
        """
        theta_new = np.zeros(self._num_theta_to_estimate)
        for i, key in enumerate(self._theta_to_estimate):
             theta_new[i] = self.dist_model[key].rvs()
        return theta_new
    
    
    def run_particle_Gibbs(self) -> None:
        """ Run particle Gibbs for parameter estimation 
        """
        # initialize likelihood, theta storage space, and chains
        WW = np.zeros(self.D)
        BB = np.zeros((self.D, self.K)).astype(int)

        theta_new = np.zeros((self.D, self._num_theta_to_estimate)) 
        chains = [Chain(model_interface=self.models_for_each_chain[d]) for d in range(self.D)]

        # draw D theta candidates for each chain, and run sMC
        for d in range(self.D):
            theta_new[d,:] = self._sample_theta_init()   
            chains[d].model_interface.update_theta(theta_new[d,:])  
            chains[d].run_particle_filter_SIR()
            chain_d = chains[d]

            model_W = chain_d.state.W
            B = chain_d._find_traj(chain_d.state.A, model_W, max=True)
            traj_X = chains[d]._get_X_traj(chain_d.state.X, B)
            BB[d,:] = B
            state_W = chain_d.model_interface.transition_model_probability(traj_X)
            WW[d] = state_W + np.log(model_W.max())
 
        
        # find optimal parameters
        W_theta = np.exp((WW - WW.max())/WW.max())
        W_theta = W_theta / W_theta.sum()

        save_std = np.zeros(self._num_theta_to_estimate)
        for p, key in enumerate(self._theta_to_estimate):
            theta_dist_mean = (theta_new[:,p]*W_theta).sum()
            theta_dist_std = np.sqrt(sum((theta_new[:,p] - theta_dist_mean)**2 * W_theta)/(self.D-1.))
            save_std[p] = theta_dist_std

            # find optimal trajectory of each chain
            self.theta_record[0,p] = theta_dist_mean


        if self.is_MAP_MCMC:
            ind_best_param = np.argmax(W_theta)
        else:

            ind_best_param = _inverse_pmf(theta_new[:,-1], W_theta, num=1)[0]

        best_model = chains[ind_best_param]
        B = BB[ind_best_param,:] 
        self.input_record[0,:] = best_model._get_R_traj(best_model.state.R, B)
        self.state_record[0,:] = best_model._get_X_traj(best_model.state.X, B)
        self.output_record[0,:] = best_model._get_Y_traj(best_model.state.Y, B)

        if self.update_theta_dist:
            best_theta = {}
            for p, key in enumerate(self._theta_to_estimate):
                best_theta[key] = [self.theta_record[0,p], save_std[p]]

            for d in range(self.D):
                chains[d].model_interface._set_parameter_distribution(update=True, theta_new=best_theta)


        # for each MCMC iteration
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
                    chains[d].model_interface.update_theta(theta_new[d,:])
                    chains[d].run_particle_filter_AS()   
                    model_W = chains[d].state.W
                    B = chains[d]._find_traj(chains[d].state.A, model_W, max=True)
                    traj_X = chains[d]._get_X_traj(chains[d].state.X, B)
                    BB[d,:] = B
                    state_W = chains[d].model_interface.transition_model_probability(traj_X)
                    WW[d] = state_W + np.log(model_W.max())
                
                W_theta = np.exp((WW - WW.max())/WW.max())
                W_theta = W_theta / W_theta.sum()

                theta_dist_mean = (theta_new[:,p]*W_theta).sum()
                theta_dist_std = np.sqrt(sum((theta_new[:,p] - theta_dist_mean)**2 * W_theta)/(self.D-1.))
                save_std[p] = theta_dist_std


                # find optimal trajectory of each chain
                self.theta_record[l+1,p] = theta_dist_mean 

            if self.is_MAP_MCMC:
                ind_best_param = np.argmax(W_theta)
            else:
                ind_best_param = _inverse_pmf(theta_new[:,p],W_theta, num=1)[0]

            best_model = chains[ind_best_param]  

            B = BB[ind_best_param,:]
            self.input_record[l+1,:] = best_model._get_R_traj(best_model.state.R, B)
            self.state_record[l+1,:] = best_model._get_X_traj(best_model.state.X, B)
            self.output_record[l+1,:] = best_model._get_Y_traj(best_model.state.Y, B)

            if self.update_theta_dist:
                best_theta = {}
                for p, key in enumerate(self._theta_to_estimate):
                    best_theta[key] = [self.theta_record[l+1,p], save_std[p]]

                for d in range(self.D):
                    chains[d].model_interface._set_parameter_distribution(update=True, theta_new=best_theta)
                

    def run_particle_Gibbs_parallel(self) -> None:
        # initialize likelihood, theta storage space, and chains
        WW = np.zeros(self.D)
        BB = np.zeros((self.D, self.K)).astype(int)

        theta_new = np.zeros((self.D, self._num_theta_to_estimate)) 
        chains = [Chain(model_interface=self.models_for_each_chain[d]) for d in range(self.D)]
        
        num_cores = mp.cpu_count()
        for d in range(self.D):
            theta_new[d,:] = self._sample_theta_init()

        args_list = [(chains[d], theta_new[d, :]) for d in range(self.D)]
        with mp.Pool(processes=num_cores) as pool:
            results = pool.starmap(worker_function_sMC, args_list)
            
        for d in range(self.D):
            WW[d], BB[d,:]= results[d]

        # find optimal parameters
        ind_best_param = np.argmax(WW)
        self.theta_record[0,:] = theta_new[ind_best_param, :]
        


        best_model = chains[ind_best_param]
        B = BB[ind_best_param,:]
        self.input_record[0,:] = best_model._get_R_traj(best_model.state.R, B)
        self.state_record[0,:] = best_model._get_X_traj(best_model.state.X, B)
        self.output_record[0,:] = best_model._get_Y_traj(best_model.state.Y, B)

        # for each MCMC iteration
        for l in tqdm(range(self.L)):
            # for each theta
            for p, key in enumerate(self._theta_to_estimate):
                theta_new = self._update_theta_at_p(
                        p=p,
                        l=l,
                        key=key
                    )

                # update chains
                args_list = [(chains[d], theta_new[d, :]) for d in range(self.D)]
                with mp.Pool(processes=num_cores) as pool:
                    results = pool.starmap(worker_function_pMCMC, args_list)

                for d in range(self.D):
                    WW[d], BB[d,:]= results[d]

                # find optimal trajectory of each chain
                ind_best_param = np.argmax(WW)  
                self.theta_record[l+1,p] = theta_new[ind_best_param, p]   
                
                best_model = chains[ind_best_param]  

                B = BB[ind_best_param,:]
                self.input_record[l+1,:] = best_model._get_R_traj(best_model.state.R, B)
                self.state_record[l+1,:] = best_model._get_X_traj(best_model.state.X, B)
                self.output_record[l+1,:] = best_model._get_Y_traj(best_model.state.Y, B)

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
        theta_temp[:,p] = self.dist_model[key].rvs(self.D)
        return theta_temp 
 
# %%
