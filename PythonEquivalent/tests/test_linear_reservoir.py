# %%
import unittest

import pandas as pd
import numpy as np

from model.model_interface import ModelInterface
from model.utils_chain import Chain

# %%
class TestLinearReservoir(unittest.TestCase):
    """Test case: Linear Reservoir

    Methods:
        run_test: test sequential Monte Carlo and particle MCMC
    """

    def runTest(self):
        # get a chopped dataframe
        df = pd.read_csv("Data/linear_reservoir.csv", index_col= 0)
        T = 50
        interval = 1
        df = df[:T:interval]
        truth = df['Q_true'].values
        # initialize model interface settings
        model_interface = ModelInterface(
            df = df,
            customized_model = None,
            theta_init = None,
            config = None,
            num_input_scenarios = 10
        )
        # initialize one chain
        chain = Chain(
            model_interface = model_interface,
            theta=[1, 0.00005]
        )

        chain.run_sequential_monte_carlo()

        B = chain._find_traj(
                chain.state.A, 
                chain.state.W
            )
        traj_sample = chain._get_X_traj(
                chain.state.X, 
                B
            )

        st_dev = np.std(traj_sample[1:] - truth, ddof = 1)
        
        self.assertLessEqual(st_dev, 0.00005, "Variance too large for sMC!!")

        # run pMCMC
        chain.run_particle_MCMC()
        
        B = chain._find_traj(
                chain.state.A, 
                chain.state.W
            )
        traj_sample = chain._get_X_traj(
                chain.state.X, 
                B
            )

        st_dev = np.std(traj_sample[1:] - truth, ddof = 1)
        
        self.assertLessEqual(st_dev, 0.00005, "Variance too large for pMCMC!!")

# %%
if __name__ == '__main__':
    unittest.main()


