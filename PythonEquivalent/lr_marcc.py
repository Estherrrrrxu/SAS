# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as ss
from typing import List

import sys

# ==========================
# model and model generation
# ==========================
def transition_model(qt: float,k: float,delta_t: float,jt: float):
    """
        give four inputs about the watershed at timestep t
        return the calculated discharge at t+1
    """
    qtp1 = (1 - k * delta_t) * qt + k * delta_t * jt
    return qtp1

# make function
def data_generation(precip: List[float], \
    delta_t: float, k: float, l: int, Q_init = 1.):
    """
        precip: input precip time series
        delta_t: time interval for precip
        k: linear reservoir factor
        l: chop the precipitation time series to N/l
        Q_init: intial discharge, default = 0. 
    """
    # chop precipitation data
    length = round(len(precip)/l)
    J = precip[:length]
    # calculate discharge using transition model
    Q = np.zeros(length+1)
    Q[0] = Q_init
    for i in range(length):
        Q[i+1] = transition_model(Q[i], k, delta_t, J[i])
    df = pd.DataFrame({"J":J,"Q":Q[1:]})
    df = df.dropna()
    return df['J'].values, df['Q'].values
# ==========================
# processes
# ==========================

# Part I: state transition
def f_theta(xht:List[float],theta: float, \
    delta_t: float, rtp1:float):
    '''
    inputs:
        xt: all possible \hat{x} at t
        theta: k in this case
        delta_t: time interval
        rtp1: input Jt with uncertainty introduced at t+1
    return:
        xtp1: \hat{x} at t+1
    '''
    xhtp1 = transition_model(xht, theta, delta_t, rtp1)
    return xhtp1

# Part II: observation
def g_theta(xht,sig_w,xt):
    """
    inputs:
        xht: all possible \hat{x} at t
        sig_w: observation uncertainty
        xt: observed x
    return:
        p(xt|sig_w, xht)
    """
    return ss.norm(xht,sig_w).pdf(xt)
# discrete inverse transform sampling
def dits(pdf,x,num):
    '''
        give x ~ pdf
    return: index of x that are been sampled according to pdf
    '''
    
    ind = np.argsort(x) # sort x according to its magnitude
    pdf = pdf[ind] # sort pdf accordingly
    cdf = pdf.cumsum()
    u = np.random.uniform(size = num)
    a = np.zeros(num)
    for i in range(num):
        if u[i] > cdf[-1]:
            a[i] = -1
        elif u[i] < cdf[0]: # this part can be removed
            a[i] = 0
        else:
            # TODO: any more efficient method?
            for j in range(1,len(cdf)):
                if (u[i] <= cdf[j]) and (u[i] > cdf[j-1]):
                    a[i] = j
                    break
    return ind[a.astype(int)]
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

def run_pGS(J,Q, delta_t, num_scenarios:int,param_samples:int, chain_len: int, sig_v0 =0.1, sig_w0=0.1,prior_mean = 0.9,prior_sd = 0.2):
    """
        for now, assume we just don't know k, and k ~ N(k0, 1)
    """
    L = chain_len
    D = param_samples
    N = num_scenarios
    k = np.zeros((D,L+1))
    k0 = np.random.normal(prior_mean, prior_sd,D)
    k[:,0] = k0
    AA = np.zeros((D+1,N,K+1)).astype(int) 
    WW = np.zeros((D,N))
    XX = np.zeros((D+1,N,K+1))
    RR = np.zeros((D+1,N,K))
    input_record = np.zeros((L,K))
    theta_record = np.zeros(L)

    for d in range(D):
        XX[d],AA[d],WW[d],RR[d] = run_sMC(J, Q, k[d,0], delta_t, N, sig_v0, sig_w0)
    
    for l in tqdm(range(L)):
        # draw new theta
        theta = np.repeat(k[:,l],N)
        posterior = WW.ravel() * ss.norm(prior_mean, prior_sd).pdf(theta)
        theta_record[l] = theta[np.argmax(posterior)]
        input_record[l] = RR[np.argmax(posterior)//10][np.argmax(posterior)%10]

        k0 = np.random.normal(prior_mean, prior_sd,D)
        k[:,l+1] = k0
        
        for d in range(D):
            XX[d],AA[d],WW[d],RR[d] = run_pMCMC(k[d,l+1], sig_v,sig_w, XX[d] , WW[d], AA[d],RR[d], J , Q, delta_t)

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

    # %%
    # ==================
    # pMCMC
    # ==================
    X, A, W, R= run_pMCMC(k, sig_v,sig_w, X, W, A,R, J, Q, delta_t)

    # %%
    # ==================
    # pGibbs
    # ==================
    num_scenarios = int(sys.argv[1])
    param_samples= int(sys.argv[2])
    chain_len= int(sys.argv[3])
    prior_mean = float(sys.argv[4])
    prior_sd = float(sys.argv[5])

    theta_record, AA, WW, XX, RR, input_record = run_pGS(J,Q, delta_t, num_scenarios,param_samples, chain_len, sig_v, sig_w, prior_mean , prior_sd )  
    
    np.savetxt(f"theta_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",theta_record,delimiter = ",")
    np.savetxt(f"input_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",input_record,delimiter = ",")
    np.savetxt(f"W_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",WW,delimiter = ",")
    np.savetxt(f"A_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",AA.reshape(AA.shape[0], -1),delimiter = ",")
    np.savetxt(f"X_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",XX.reshape(XX.shape[0], -1),delimiter = ",")


# %%


# %%
# Add isotope
# C_old = 1000
# C_J = C_old + np.random.randn(length)*1000.0
# C_Q = np.zeros(length)
# kappa = k*delta_t
# # j - timestep
# # i - age step

# # for all timesteps
# for t in range(length):
#     # now the maximum age in the system is t
#     pq = np.zeros(t+2) # store cq that goes away            
#     # when T = 0
#     pq[0] = (kappa+np.exp(-kappa)-1)*J[t]/kappa/delta_t/Q[t]
#     # get C_Q at T == 0
#     C_Q[t] += C_J[t]*pq[0]*delta_t

#     # when T > 0
#     for T in range(1,t+1):
#         # calculate SAS
#         pq[T] = (1-np.exp(-kappa)*J[t-T])               
#         # get C_Q
#         C_Q[t] += C_J[t-T]*pq[T]*delta_t
#     C_Q[t] += C_old*(1-pq.sum()*delta_t)