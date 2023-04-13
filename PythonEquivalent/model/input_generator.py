import numpy as np

# %%

class InputGenerator:
    def __init__(
        self,
        theta_input: float,
    ):
        pass

    def generate_random_uniform(self,
        ) -> np.ndarray:
        """Generate input uncertainty for the model
        
        Returns:
            np.ndarray: input for the model"""   

        self.R = np.zeros((self.T, self.N))
        for k in range(self.T):
            self.R[k,:] = self.model.input_model(self.influx[k])
        return
    def get_gaussian_process(self,
        ) -> np.ndarray:
        pass

    def input_model(self, Ut:float) -> np.ndarray:
        """Input model for linear reservoir

        Rt = Ut - U(0, U_upper)

        Args:
            Ut (float): forcing at time t

        Returns:
            np.ndarray: Rt
        """
        return ss.uniform(Ut-self.theta_input, 
                          self.theta_input
                          ).rvs(self.N)
                             