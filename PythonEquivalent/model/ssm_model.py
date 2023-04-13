# %%
from input_generator import InputGenerator
from model_interface import ModelInterface
from utils_chain import Chain

# %%
class SSModel:
    def __init__(
        self,
        input_generator: InputGenerator,
        user_interface: ModelInterface,
        df: pd.DataFrame,
        num_input_scenarios: Optional[int] = 10,
    ) -> None:
        
        """
        Args:
        instance: 
        resample theta
        then update model
        need to call chain object
        """
        self.input_generator = input_generator
        self.model = model
        self.model_links = model_links

# TODO: call inputs after input model


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

    def sample_theta_from_prior(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        theta_new = np.zeros(self._num_theta_to_estimate)
        for i, key in enumerate(self._theta_to_estimate):
             theta_new[i] = self.prior_model[key].rvs()
        # cannot update model here, because it need to update iteratively
        self._update_model(theta_new)
        return theta_new

 