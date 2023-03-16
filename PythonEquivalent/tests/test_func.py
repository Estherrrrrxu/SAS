# %%
import unittest
import pandas as pd
from functions.utils import plot_MLE
from model import run_sMC, run_pMCMC
import numpy as np
# %%
class TestFuncs:
    def __init__(self, filename = "Dataset.csv", T = 50, interval = 1, k = 1,delta_t = 1./24/60*15,\
                 theta_ipt = 0.254*1./24/60*15,theta_obs = 0.00005,N = 50):
        df = pd.read_csv(filename, index_col= 0)
        self.T = T
        self.interval = interval
        self.df = df[:self.T]
        self.J_obs = self.df['J_obs'][::self.interval].values
        self.Q_obs = self.df['Q_obs'][::self.interval].values
        self.K = len(self.J_obs)
        self.delta_t = delta_t
        self.delta_t *= self.interval
        # estimation inputs
        self.N = N
        self.theta = {'k': k, 'sig_v': theta_ipt, 'sig_w': theta_obs*10}

    # ==================
    # sMC
    # ==================
    def test_sMC(self): 
        """
        Test if sMC is working as expected
        """
        self.X, self.A, self.W, self.R = run_sMC(self.J_obs, self.Q_obs, self.theta, self.delta_t,self.N)
        MLE = plot_MLE(self.X, self.A, self.W, self.R,self.K,self.df,self.J_obs, self.Q_obs,left = 0, right = 30)
        truth = self.df['Q_true']
        sum_squared_residuals = sum((MLE[1:] - truth)**2)
        return np.sqrt(sum_squared_residuals/self.K)
    # %%
    # ==================
    # pMCMC
    # ==================
    def test_pMCMC(self): 
        """
        Test if pMCMC is working as expected
        """
        self.X, self.A, self.W, self.R = run_pMCMC(self.theta, self.X, self.W, self.A,self.R, self.J_obs,self.Q_obs, self.delta_t)
        MLE = plot_MLE(self.X, self.A, self.W, self.R,self.K,self.df,self.J_obs, self.Q_obs,left = 0, right = 30)
        truth = self.df['Q_true']
        sum_squared_residuals = sum((MLE[1:] - truth)**2)
        return np.sqrt(sum_squared_residuals/self.K)

class TestTwoFunctions(unittest.TestCase):
    def runTest(self):
        result_std =   TestFuncs(filename = "Dataset.csv", T = 50, interval = 1, k = 1,delta_t = 1./24/60*15,\
                    theta_ipt = 0.254*1./24/60*15,theta_obs = 0.00005,N = 50)
        assert result_std.test_sMC() <= result_std.theta['sig_w']/10, "Variance too large for sMC!!"
        assert result_std.test_pMCMC() <= result_std.theta['sig_w']/10, "Variance too large for pMCMC!!"


if __name__ == '__main__':
    unittest.main()


