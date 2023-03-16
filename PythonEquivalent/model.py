# %%
import pandas as pd
import numpy as np
from typing import List
import scipy.stats as ss
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

"""
    Considering the following condition:
    Q = k * S
    J - Q = dS/dT
    Goal: use particle filter to estimate theta
"""
from functions.process import f_theta, g_theta
from functions.utils import dits

# %%   
# ==========================
# MODEL part
# ==========================
def run_sMC(J: List[float], Q: List[float], theta: dict, delta_t: float,N:int):
    '''
        definitions same as the wrapper
    return: qh  - estiamted state in particles
            P   - weight associated with each particles
            A   - ancestor lineage
            M   - total number of particles
    '''
    sig_v = theta['sig_v']
    sig_w = theta['sig_w']
    k = theta['k']
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
    W = np.log(np.ones(N)/N) # initial weight on particles are all equal
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
        wkp1 = wk + np.log(g_theta(xkp1, sig_w, Q[kk]))
        W = wkp1
        X[:,kk+1] = xkp1
        aa = dits(W,A[:,kk], num = N)
        A[:,kk+1] = aa        
    return X,A,W,R


def run_pMCMC(theta:dict, X: List[float], W: List[float], A: List[float],R: List[float], J: List[float] , Q: List[float], delta_t: float):
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
    sig_v = theta['sig_v']
    sig_w = theta['sig_w']
    k = theta['k']

    X = X.copy()
    W = W.copy()
    A = A.copy()
    R = R.copy()
    # sample an ancestral path based on final weight
    K = len(Q)
    N = X.shape[0]
    B = np.zeros(K+1).astype(int)
    B[-1] = dits(W,A[:,-1], num = 1)
    for i in reversed(range(1,K+1)):
        B[i-1] =  A[:,i][B[i]]
    # state estimation-------------------------------
    W = np.log(np.ones(N)/N) # initial weight on particles are all equal
    for kk in range(K):
    # for kk in tqdm(range(K), desc ="PMCMC"):
        # compute new state and weights based on the model
        # xkp1 = f_theta(X[:,kk],k,delta_t,J[kk],sig_v).rvs()
        # rr = ss.uniform(J[kk]-sig_v,sig_v).rvs(N)
        rr = ss.uniform(J[kk]-sig_v,sig_v).rvs(N)
        xkp1 = f_theta(X[:,kk],k,delta_t,rr)
        # Look into this
        x_prime = X[B[kk+1],kk+1]
        W_tilde = W + np.log(ss.norm(x_prime,0.000005).pdf(xkp1))
        # ^^^^^^^^^^^^^^
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
        wkp1 = W + np.log(g_theta(xkp1, sig_w, Q[kk]))
        W = wkp1#/wkp1.sum()

    return X, A, W, R

def run_pMH(J,Q, delta_t, num_scenarios:int,num_chains:int, chain_len: int, sig_v =0.1, sig_w=0.1,prior_mean = 0.9,prior_sd = 0.2, theta_step = -1, max_rejections = 10):
    """
        for now, assume we just don't know k, and k ~ N(k0, 1)
    """

    def get_theta_proposal(theta_0, theta_step):
        theta_proposal = np.random.normal(theta_0, theta_step)
        forward_step_probability = ss.norm(theta_0, theta_step).pdf(theta_proposal)
        backward_step_probability = ss.norm(theta_proposal, theta_step).pdf(theta_0)
        return theta_proposal, forward_step_probability, backward_step_probability

    L = chain_len
    D = num_chains
    N = num_scenarios
    K = len(J)
    kk = np.zeros((D,L+1))
    AA = np.zeros((D+1,L+1,N,K+1)).astype(int) 
    WW = np.zeros((D,L+1,N))
    XX = np.zeros((D+1,L+1,N,K+1))
    RR = np.zeros((D+1,L+1,N,K))
    AA_q = np.zeros((D+1,N,K+1)).astype(int) 
    WW_q = np.zeros((D,N))
    XX_q = np.zeros((D+1,N,K+1))
    RR_q = np.zeros((D+1,N,K))
    #input_record = np.zeros((L,K))
    #theta_record = np.zeros((L,D))
    # theta_record = np.zeros(L+1)
    # theta_record[0] = prior_mean
    pbar = tqdm(total = D+1)
    print('starting')
    for d in range(D):
        pbar.update(1)
        print('')
        print(f'{d=}')
        
        rejection_count = np.inf
        while rejection_count>max_rejections:
            rejection_count = 0
            ll = 0
            goodstart=False
            while not goodstart:
                kk[d,ll] = np.random.normal(prior_mean, prior_sd,1)
                print(f'starting run_sMC')
                XX[d,ll,:,:],AA[d,ll,:,:],WW[d,ll,:],RR[d,ll,:,:] = run_sMC(J, Q, kk[d,0], delta_t, N, sig_v, sig_w)
                print(f'done run_sMC')
                if WW[d,ll,:].max()>-np.inf:
                    goodstart = True

        
            print(f'{WW[d,ll,:].sum()=}')
            while ll<L:
                # =============================
                # M-H
                theta_0 = kk[d,ll].copy()
                theta_proposal, forward_step_probability, backward_step_probability = get_theta_proposal(theta_0, theta_step)
                current_prior_probability = ss.norm(prior_mean, prior_sd).pdf(theta_0)
                proposal_prior_probability = ss.norm(prior_mean, prior_sd).pdf(theta_proposal)
                print(f'starting run_pMCMC')
                XX_q,AA_q,WW_q,RR_q = run_pMCMC(theta_proposal, sig_v,sig_w, XX[d,ll,:,:] , WW[d,ll,:], AA[d,ll,:,:],RR[d,ll,:,:], J , Q, delta_t)
                print(f'done run_pMCMC')
                print(f'{WW_q.max()=}')
                print(f'{WW[d,ll,:].max()=}')
                posterior_q = np.exp(np.max(WW_q))
                posterior_0 = np.exp(np.max(WW[d,ll,:]))
                # alpha = np.prod(posterior_q * ss.norm(theta_0, width).pdf(theta_q)) / np.prod(posterior_0 * ss.norm(theta_q, width).pdf(theta_0))
                alpha = (posterior_q * proposal_prior_probability / forward_step_probability) / (posterior_0 * current_prior_probability / backward_step_probability)
                # alpha = (posterior_q  / forward_step_probability) / (posterior_0 / backward_step_probability)

                u = np.random.uniform()
                if u <= alpha:
                    XX[d,ll+1,:,:], AA[d,ll+1,:,:], WW[d,ll+1,:], RR[d,ll+1,:,:] = XX_q, AA_q, WW_q, RR_q
                    kk[d,ll+1] = theta_proposal
                    ll+=1
                    rejection_count = 0
                    print(f'{theta_proposal=} accepted')
                else:
                    rejection_count += 1
                    print(f'{theta_proposal=} rejected')
                if rejection_count>max_rejections:
                    break
    pbar.close()
    theta = kk

    return theta, AA,WW, XX,RR#, input_record

def run_pGS_SAEM(J,Q, delta_t, num_scenarios:int,num_chains:int, chain_len: int, theta_init:dict, q_step = -1):
    """
        for now, assume we just don't know k, and k ~ N(k0, 1)
    """
    def prior_model(theta_model,D):
        kk = np.random.normal(theta_model['k']['prior_mean'],theta_model['k']['prior_sd'],D)
        sig_ww = np.random.uniform(theta_model['sig_w']['lower'],theta_model['sig_w']['upper'],D)
        return kk, sig_ww
    def update_model(theta_model,theta_record_ll,D):
        kk = np.random.normal(theta_record_ll[0],theta_model['k']['search_range'],D)
        sig_ww = np.random.normal(theta_record_ll[1],theta_model['sig_w']['search_range'],D)
        return kk, sig_ww

    sig_v = theta_init['sig_v']['val']

    L = chain_len
    D = num_chains
    N = num_scenarios
    K = len(J)
    kk = np.zeros((D,L+1))
    sig_w = np.zeros((D,L+1))
    AA = np.zeros((D+1,L+1,N,K+1)).astype(int) 
    WW = np.zeros((D,L+1,N))
    XX = np.zeros((D+1,L+1,N,K+1))
    RR = np.zeros((D+1,L+1,N,K))
    #input_record = np.zeros((L,K))
    #theta_record = np.zeros((L,D))
    theta_record = np.zeros((2,L+1))
    theta_record[:,0] = [theta_init['k']['prior_mean'],theta_init['sig_w']['upper']]
    

    ll = 0
    kk[:,ll],sig_w[:,ll] = prior_model(theta_init,D)
    for d in range(D):
        theta_step = {'k':kk[d,0], 'sig_w':sig_w[d,0],'sig_v': sig_v}
        XX[d,ll,:,:],AA[d,ll,:,:],WW[d,ll,:],RR[d,ll,:,:] = run_sMC(J, Q, theta_step, delta_t, N)

    Qh = q_step[0] * np.max(WW[:,ll,:],axis = 1)
    for ll in tqdm(range(L)):
        # generate random variables now
        theta_record_ll = theta_record[:,ll]        
        theta_proposal_k, theta_proposal_sig_w = update_model(theta_init,theta_record_ll,D)
        # ======update=========

        # Param 1
        for d in range(D):
            theta_step = {'k':theta_proposal_k[d], 'sig_w':theta_record_ll[1],'sig_v': sig_v}
            XX[d,ll+1,:,:], AA[d,ll+1,:,:], WW[d,ll+1,:], RR[d,ll+1,:,:] = run_pMCMC(theta_step, XX[d,ll,:,:] , WW[d,ll,:], AA[d,ll,:,:],RR[d,ll,:,:], J , Q, delta_t)
            print(sum(WW[d,ll+1,:]))
        Qh = (1-q_step[ll+1])*Qh + q_step[ll+1] * np.max(WW[:,ll+1,:],axis = 1)
        theta_record[0,ll+1] = theta_proposal_k[np.argmax(Qh)]
        
        # Param 2
        for d in range(D):
            theta_step = {'k':theta_record[0,ll+1], 'sig_w':theta_proposal_sig_w[d],'sig_v': sig_v}
            XX[d,ll+1,:,:], AA[d,ll+1,:,:], WW[d,ll+1,:], RR[d,ll+1,:,:] = run_pMCMC(theta_step, XX[d,ll+1,:,:] , WW[d,ll+1,:], AA[d,ll+1,:,:],RR[d,ll+1,:,:], J , Q, delta_t)

        Qh = (1-q_step[ll+1])*Qh + q_step[ll+1] * np.max(WW[:,ll+1,:],axis = 1)
        theta_record[1,ll+1] = theta_proposal_sig_w[np.argmax(Qh)]




    return theta_record, AA,WW, XX,RR#, input_record


# TODO: the update rate of states; the correlation length of the Markov chain; and the convergence of the marginal posteriors of the parameters
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
    # import os
    # os.chdir('./PythonEquivalent/')
    df = pd.read_csv("Dataset.csv", index_col= 0)
    T = 50
    interval = 1
    df = df[:T]
    J_obs = df['J_obs'][::interval]
    Q_obs = df['Q_obs'][::interval]
    #plot_input(df,J_obs, Q_obs)
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
    sig_w = 0.0005
    #sig_w = theta_obs

    # %%
    # ==================
    # pGibbs
    # ==================
    num_scenarios = 50
    param_samples = 10
    chain_len = 100
    prior_mean_k = 1.2
    prior_sd_k = 0.3
    search_k = 0.05
    lower_w = sig_w/10
    upper_w = sig_w
    search_w = lower_w/5
    # q_step = 0.7
    q_step = np.ones(chain_len+1)*0.75
    # q_step[:40] *= 0.7
    # q_step[40:] *= 0.9
    # TODO: define the param to learn
    theta_init = {'sig_v': {'val': sig_v}, \
                'sig_w':{'lower': lower_w,'upper':upper_w,'search_range':search_w},\
                'k':{'prior_mean': prior_mean_k,'prior_sd':prior_sd_k,'search_range':search_k},\
            }

    theta_record, AA, WW, XX, RR = run_pGS_SAEM(J,Q, delta_t, num_scenarios,param_samples, chain_len, theta_init, q_step)  
    # %%
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(theta_record[0])
    plt.ylabel(r"$\theta$")
    plt.xlabel("MCMC Chain")
    plt.subplot(2,1,2)
    plt.plot(theta_record[1])
    plt.ylabel(r"$\theta$")
    plt.xlabel("MCMC Chain")
    # plt.plot([0,chain_len],[theta_record.T[10:].mean(),theta_record.T[10:].mean()],'r',label = "prior")
    plt.legend()

    # %%
    # not to worry about MH for now
    # num_scenarios = 2
    # param_samples = 1
    # chain_len = 50
    # prior_mean = 0.8
    # prior_sd = 0.3
    # theta_step = 0.05
    # # ns = int(sys.argv[1])
    # # ps = int(sys.argv[2])
    # # cl = int(sys.argv[3])
    # # mean = float(sys.argv[4])
    # # sd = float(sys.argv[5])
    # theta_record, AA, WW, XX, RR = run_pMH(J,Q, delta_t, num_scenarios,param_samples, chain_len, sig_v, sig_w, prior_mean , prior_sd, theta_step,max_rejections=10) 
    # plt.figure()
    # plt.plot(theta_record.T)
    # plt.ylabel(r"$\theta$")
    # plt.xlabel("MCMC Chain")
    # plt.plot([0,chain_len],[theta_record.T[10:].mean(),theta_record.T[10:].mean()],'r',label = "prior")

    # %%
    plt.figure()
    plt.plot(theta_record.T)
    plt.ylabel(r"$\theta$")
    plt.xlabel("MCMC Chain")
    plt.figure()
    plt.hist(theta_record.T,bins = 10,label = "theta_distribution")
    x = np.linspace(ss.norm(prior_mean,prior_sd).ppf(0.01),ss.norm(prior_mean,prior_sd).ppf(0.99), 100)
    plt.plot(x,ss.norm(prior_mean,prior_sd).pdf(x),label = "prior")
    plt.xlabel(r"$\theta$")
    plt.legend()

    # %%
    plt.figure()
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    for d in range(param_samples):
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
        ax1.plot(df['J_true'],label = "J true")
        ax1.plot(J_obs,"*",label = "J obs")
        #ax1.plot(input_record.T,'r.',alpha = 0.01)
        ax1.legend(frameon = False)
        ax1.set_xlim([left,right])
        ax1.set_title('J')

        ax2.plot(Q_obs,".", label = "Observation")
        ax2.plot(df['Q_true'],'k', label = "Hidden truth")
        ax2.plot(np.linspace(0,T,len(MLE)-1),MLE[1:],'r:', label = "One Traj/MLE")
        ax2.legend(frameon = False)
        ax2.set_title(f"sig_v {round(sig_v,5)}, sig_w {sig_w}")
        ax2.set_xlim([left,right])
        ax2.tight_layout()

    print('done')




     # %%
    # np.savetxt(f"Results/theta_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",theta_record,delimiter = ",")
    # np.savetxt(f"Results/input_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",input_record,delimiter = ",")
    # np.savetxt(f"Results/W_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",WW,delimiter = ",")
    # np.savetxt(f"Results/A_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",AA.reshape(AA.shape[0], -1),delimiter = ",")
    # np.savetxt(f"Results/X_{num_scenarios}_{param_samples}_{chain_len}_{prior_mean}_{prior_sd}.csv",XX.reshape(XX.shape[0], -1),delimiter = ",")


# %%
