# %%
import unittest

import pandas as pd

import matplotlib.pyplot as plt
from functions.link import ModelLink
from functions.estimator import SSModel

from useless.utils import plot_MLE
# This test need to be properly set up later!!
# %%


class TestLinearReservoir(unittest.TestCase):

    def runTest(self):
        # get a chopped dataframe
        df = pd.read_csv("Dataset.csv", index_col= 0)
        T = 50
        interval = 1
        df = df[:T:interval]
        # %%
        model_link = ModelLink(df=df, num_input_scenarios=15)
        default_model = SSModel(model_link)

        state = default_model.run_sequential_monte_carlo([1.,0.00005])
        plot_MLE(state,df,left = 0, right = 50)

        state = default_model.run_particle_MCMC(state,theta = [1.,0.00005])
        plot_MLE(state,df,left = 0, right = 50)

        result_std =  TestClass(filename = "Dataset.csv", T = 50, interval = 1, k = 1,delta_t = 1./24/60*15,\
                    theta_ipt = 0.254*1./24/60*15,theta_obs = 0.00005,N = 50)
        
        assert result_std.test_sMC() <= result_std.theta['sig_w']/10, "Variance too large for sMC!!"
        assert result_std.test_pMCMC() <= result_std.theta['sig_w']/10, "Variance too large for pMCMC!!"



# %%



if __name__ == '__main__':
    unittest.main()


