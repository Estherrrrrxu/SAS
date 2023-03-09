# %%
import pandas as pd
import numpy as np
from typing import List
import scipy.stats as ss
from tqdm import tqdm
import sys

"""
    Considering the following condition:
    Q = k * S
    J - Q = dS/dT
    Goal: use particle filter to estimate theta
"""
from functions.process import f_theta, g_theta
from functions.utils import dits,plot_input,plot_MLE

# %%   
# ==========================
# MODEL part
# ==========================
def run_sMC(J: List[float], Q: List[float], k: float, delta_t: float,N:int, sig_v: float, sig_w: float):
    '''
        definitions same as the wrapper
    return: qh  - estiamted state in particles
            P   - weight associated with each particles
            A   - ancestor lineage
            M   - total number of particles
    '''
    # initialization---------------------------------
    K = len(J)
    A = np.zeros((N,K+1)).astype(int) # ancestor storage
    A[:,0] = np.arange(N) # initialize the first set of particles
    # state storage, m_Q
    # X = np.zeros((N,T+1)) # for each particle, for each X
    X = np.zeros((N,K+1)) # for each particle, for each X
    # initialize X0 by giving one value to the model
    # assume X0 = 0 --> Dirac Delta distribution at 0
    X[:,0] = np.ones(N)*Q[0] # for each particle, for each X
    # and we only need to store the last weight
    W = np.ones(N)/N # initial weight on particles are all equal
    R = np.zeros((N,K)) # store the stochasity from input concentration
    # state estimation---------- ---------------------
    for kk in range(K):
    # for kk in tqdm(range(K), desc ="sMC"):
        # draw new state samples and associated weights based on last ancestor
        xk = X[A[:,kk],kk]
        wk = W

        # compute new state and weights based on the model
        # xkp1 = f_theta(xk,k,delta_t,J[kk],sig_v).rvs()
        R[:,kk] = ss.uniform(J[kk]-sig_v,sig_v).rvs(N)
        xkp1 = f_theta(xk,k,delta_t,R[:,kk])
        wkp1 = wk * g_theta(xkp1, sig_w, Q[kk])
        wkp1 = wkp1/wkp1.sum() # normalize
        W = wkp1
        X[:,kk+1] = xkp1
        aa = dits(W,A[:,kk], num = N)
        A[:,kk+1] = aa        
    return X,A,W,R


def run_pMCMC(k: float, sig_v: float,sig_w: float, X: List[float], W: List[float], A: List[float],R: List[float], J: List[float] , Q: List[float], delta_t: float):
    '''
    pMCMC inside loop, run this entire function as many as possible
    update w/ Ancestor sampling
        theta           - let it be k, sig_v, sig_w for now
        nstep           - number of steps in MCMC
        J,Q,k,delta_t   - same as defined above
        For reference trajectory:
        qh              - estiamted state in particles
        P               - weight associated with each particles
        A               - ancestor lineage
        
    '''
    # sample an ancestral path based on final weight
    K = len(Q)
    N = X.shape[0]
    B = np.zeros(K+1).astype(int)
    B[-1] = dits(W,A[:,-1], num = 1)
    for i in reversed(range(1,K+1)):
        B[i-1] =  A[:,i][B[i]]
    # state estimation-------------------------------
    for kk in range(K):
    # for kk in tqdm(range(K), desc ="PMCMC"):
        # compute new state and weights based on the model
        # xkp1 = f_theta(X[:,kk],k,delta_t,J[kk],sig_v).rvs()
        # rr = ss.uniform(J[kk]-sig_v,sig_v).rvs(N)
        rr = ss.uniform(J[kk]-sig_v,sig_v).rvs(N)
        xkp1 = f_theta(X[:,kk],k,delta_t,rr)
        x_prime = X[B[kk+1],kk+1]
        W_tilde = W * ss.norm(x_prime,0.000005).pdf(xkp1)
        W_tilde = W_tilde/W_tilde.sum()
        A[B[kk+1],kk+1] = dits(W_tilde,xkp1 - x_prime, num = 1)
        # now update everything in new state
        notB = np.arange(0,N)!=B[kk+1]
        A[:,kk+1][notB] = dits(W,X[:,kk], num = N-1)
        xkp1[notB] = xkp1[A[:,kk+1][notB]]
        rr[notB] = rr[A[:,kk+1][notB]]
        xkp1[~notB] = xkp1[A[B[kk+1],kk+1]]
        rr[~notB] = R[:,kk][A[B[kk+1],kk+1]]
        X[:,kk+1] = xkp1   
        R[:,kk] = rr       
        W[notB] = W[A[:,kk+1][notB]]
        W[~notB] = W[A[B[kk+1],kk+1]]
        wkp1 = W * g_theta(xkp1, sig_w, Q[kk])
        W = wkp1/wkp1.sum()

    return X, A, W, R

def run_pGS(J,Q, delta_t, num_scenarios:int,param_samples:int, chain_len: int, sig_v =0.1, sig_w=0.1,prior_mean = 0.9,prior_sd = 0.2):
    """
        for now, assume we just don't know k, and k ~ N(k0, 1)
    """
    L = chain_len
    D = param_samples
    N = num_scenarios
    kk = np.zeros((D,L+1))
    k0 = np.random.normal(prior_mean, prior_sd,D)
    kk[:,0] = k0
    AA = np.zeros((D+1,N,K+1)).astype(int) 
    WW = np.zeros((D,N))
    XX = np.zeros((D+1,N,K+1))
    RR = np.zeros((D+1,N,K))
    input_record = np.zeros((L,K))
    theta_record = np.zeros(L)
    # theta_record = np.zeros(L+1)
    # theta_record[0] = prior_mean

    for d in range(D):
        XX[d],AA[d],WW[d],RR[d] = run_sMC(J, Q, kk[d,0], delta_t, N, sig_v, sig_w)
    
    for ll in tqdm(range(L)):
        posterior = np.prod(WW, axis = 1) * ss.norm(prior_mean, prior_sd).pdf(kk[:,ll])
        theta_record[ll] = kk[:,ll][np.argmax(posterior)]
        # this will be used after fix A's problem
        # input_record[ll] = RR[np.argmax(posterior),N,:]

        k0 = np.random.normal(prior_mean, prior_sd,D)
        kk[:,ll+1] = k0
        
        for d in range(D):
            XX[d],AA[d],WW[d],RR[d] = run_pMCMC(kk[d,ll+1], sig_v,sig_w, XX[d] , WW[d], AA[d],RR[d], J , Q, delta_t)

        # # draw new theta
        # multiplier = ss.norm(theta_record[ll], prior_sd/5).pdf(np.repeat(kk[:,ll],N))
        # posterior = WW.ravel() * multiplier
        # theta_record[ll+1] = kk[:,ll][np.argmax(posterior)//N]  
        # input_record[ll] = RR[np.argmax(posterior)//N][np.argmax(posterior)%N]

        # k0 = ss.norm(theta_record[ll], prior_sd/5).rvs(D)
        # kk[:,ll+1] = k0
        
        # for d in range(D):
        #     XX[d],AA[d],WW[d],RR[d] = run_pMCMC(kk[d,ll+1], sig_v,sig_w, XX[d] , WW[d], AA[d],RR[d], J , Q, delta_t)

    return theta_record, AA,WW, XX,RR, input_record

# %%
# ======================================================
# Run this part
# ======================================================
if __name__ == "__main__":
    """
        J: precip
        Q: discharge
        k: linear reservoir factor
        delta_t: timestep
        sig_q: importance sampling on state transition
        sig_v: noise on transition process
        sig_w: noise on observation process
        M: number of particles
    """
    # ==================
    # Get data
    # ==================
    df = pd.read_csv("Dataset.csv", index_col= 0)
    T = 500
    interval = 1
    df = df[:T]
    J_obs = df['J_obs'][::interval]
    Q_obs = df['Q_obs'][::interval]
    plot_input(df,J_obs, Q_obs)
    # ==================
    # Set training data
    # ==================
    J = J_obs.values
    Q = Q_obs.values
    # true model inputs
    k = 1
    delta_t = 1./24/60*15
    theta_ipt = 0.254*delta_t
    theta_obs = 0.00005
    K = len(J)

    delta_t *= interval
    # estimation inputs
    sig_v = theta_ipt
    sig_w = theta_obs
    N = 50
    # %%
    # ==================
    # sMC
    # ==================
    X, A, W, R = run_sMC(J, Q, k, delta_t,N,sig_v,sig_w)
    plot_MLE(X,A,W,R,K,df,J_obs, Q_obs,sig_v,sig_w,left = 0, right = 30)

    # %%
    # ==================
    # pMCMC
    # ==================
    X, A, W, R= run_pMCMC(k, sig_v,sig_w, X, W, A,R, J, Q, delta_t)
    plot_MLE(X,A,W,R,K,df,J_obs, Q_obs,sig_v,sig_w,left = 0, right = 500)



    # %%
    # ==================
    # pGibbs
    # ==================
    num_scenarios = 20
    param_samples= 20
    chain_len = 50
    prior_mean = 0.9
    prior_sd = 0.2
    # ns = int(sys.argv[1])
    # ps = int(sys.argv[2])
    # cl = int(sys.argv[3])
    # mean = float(sys.argv[4])
    # sd = float(sys.argv[5])
    theta_record, AA, WW, XX, RR, input_record = run_pGS(J,Q, delta_t, num_scenarios,param_samples, chain_len, sig_v, sig_w, prior_mean , prior_sd )  
    # %%
    plt.figure()
    plt.plot(theta_record)
    plt.ylabel(r"$\theta$")
    plt.xlabel("MCMC Chain")
    plt.figure()
    plt.hist(theta_record,bins = 10,label = "theta_distribution")
    x = np.linspace(ss.norm(prior_mean,prior_sd).ppf(0.01),ss.norm(prior_mean,prior_sd).ppf(0.99), 100)
    plt.plot(x,ss.norm(prior_mean,prior_sd).pdf(x),label = "prior")
    plt.xlabel(r"$\theta$")
    plt.legend()

    # %%
    id = np.argmax(np.nan_to_num(WW[:,-1],0))

    left = 0
    right = 500

    B = np.zeros(K+1).astype(int)
    B[-1] = AA[-2,:,-1][id]
    for i in reversed(range(1,K+1)):
        B[i-1] =  AA[-2,:,i][B[i]]
    MLE = np.zeros(K+1)
    for i in range(K+1):
        MLE[i] = XX[-2,B[i],i]
    # ------------
    # ------------
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(df['J_true'],label = "J true")
    plt.plot(J_obs,"*",label = "J obs")
    plt.plot(input_record.T,'r.',alpha = 0.01)
    plt.legend(frameon = False)
    plt.xlim([left,right])
    plt.title('J')

    plt.subplot(2,1,2)
    plt.plot(Q_obs,".", label = "Observation")
    plt.plot(df['Q_true'],'k', label = "Hidden truth")
    plt.plot(np.linspace(0,T,len(MLE)-1),MLE[1:],'r:', label = "One Traj/MLE")
    plt.legend(frameon = False)
    plt.title(f"sig_v {round(sig_v,5)}, sig_w {sig_w}")
    plt.xlim([left,right])
    plt.tight_layout()





     # %%
    np.savetxt(f"Results/theta_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",theta_record,delimiter = ",")
    np.savetxt(f"Results/input_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",input_record,delimiter = ",")
    np.savetxt(f"Results/W_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",WW,delimiter = ",")
    np.savetxt(f"Results/A_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",AA.reshape(AA.shape[0], -1),delimiter = ",")
    np.savetxt(f"Results/X_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",XX.reshape(XX.shape[0], -1),delimiter = ",")


# %%
