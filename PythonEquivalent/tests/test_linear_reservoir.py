# %%
import unittest

import pandas as pd
import numpy as np

from functions.link import ModelLink
from functions.estimator import SSModel

# %%
class TestLinearReservoir(unittest.TestCase):
    """Test case: Linear Reservoir

    Methods:
        run_test: test sequential Monte Carlo and particle MCMC
    """

    def runTest(self):
        # get a chopped dataframe
        df = pd.read_csv("Data/linear_reservoir.csv", index_col = 0)
        T = 50
        interval = 1
        df = df[:T:interval]
        # initialize the model
        model_link = ModelLink(df = df, 
                               num_input_scenarios = 15)
        default_model = SSModel(model_link)
        # get the truth
        truth = default_model.model_link.df['Q_true']
        
        # run sMC
        state = default_model.run_sequential_monte_carlo([1., 0.00005])
        B = default_model._find_traj(state.A, state.W)
        traj_sample = default_model._get_X_traj(state.X, B)

        st_dev = np.std(traj_sample[1:] - truth, ddof = 1)
        
        self.assertLessEqual(st_dev, 0.00005, "Variance too large for sMC!!")

        # run pMCMC
        state = default_model.run_particle_MCMC(state,theta = [1.,0.00005])
        B = default_model._find_traj(state.A, state.W)
        traj_sample = default_model._get_X_traj(state.X, B)

        st_dev = np.std(traj_sample[1:] - truth, ddof = 1)
        
        self.assertLessEqual(st_dev, 0.00005, "Variance too large for pMCMC!!")

# %%
if __name__ == '__main__':
    unittest.main()


