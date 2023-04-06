# %%
import unittest

import pandas as pd
import numpy as np

from functions.link import ModelLink
from functions.estimator import SSModel

# %%
class TestDoubleWell(unittest.TestCase):
    """Test case: Double Well potential

    Methods:
        run_test: test sequential Monte Carlo and particle MCMC
    """
    def create_input_time_series(self):
        """Create input time series
        
        Consider Ito diffusion w/ drift:
            dX_t = b_\theta(X_t)dt + \sigma() 
        """
        # create a double well potential
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = (X**2 - 1)**2 + Y**2
        # create a dataframe
        df = pd.DataFrame(columns = ['X', 'Y', 'Q_true'])
        df['X'] = X.reshape(-1)
        df['Y'] = Y.reshape(-1)
        df['Q_true'] = Z.reshape(-1)
        # save the dataframe
        df.to_csv("Dataset.csv")

    def runTest(self):
        # get a chopped dataframe
        df = pd.read_csv("Dataset.csv", index_col= 0)
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

        state = default_model.run_particle_MCMC(state,theta = [1.,0.00005])
        B = default_model._find_traj(state.A, state.W)
        traj_sample = default_model._get_X_traj(state.X, B)

        st_dev = np.std(traj_sample[1:] - truth, ddof = 1)
        
        self.assertLessEqual(st_dev, 0.00005, "Variance too large for pMCMC!!")

# %%
if __name__ == '__main__':
    unittest.main()


