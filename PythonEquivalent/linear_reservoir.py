# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from typing import List
# %%
"""
    Considering the following condition:
    Q = k * S
    J - Q = dS/dT
    Goal: use particle filter to estimate theta
"""
def transition_model(qt: float,k: float,delta_t: float,jt: float):
    """
        give four inputs about the watershed at timestep t
        return the calculated discharge at t+1
    """
    qtp1 = (1 - k * delta_t) * qt + k * delta_t * jt
    return qtp1
# %%
# make function
def data_generation(precip: List[float], \
    delta_t: float, k: float, l: int, Q_init = 0, plot = True):
    """
        precip: input precip time series
        delta_t: time interval for precip
        k: linear reservoir factor
        l: chop the precipitation time series to N/l
        Q_init: intial discharge, default = 0. 
    """
    # adjust k according to time step
    k /= delta_t
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
    if plot == True:
        plt.figure()
        plt.subplot(4,1,1)
        plt.plot(df['J'])
        plt.title('J')
        plt.subplot(4,1,2)
        plt.plot(df['Q'])
        plt.title('Q')
        plt.tight_layout()
    return df['J'].values, df['Q'].values

# %%   
# MODEL part
# define utility functions---------------------
# state transition
def f_theta(qht:List[float],k: float, \
    delta_t: float, jt:float, sig_v: float):
    '''
        qt: all possible \hat{x}_{t-1}
    return: qhtp1
    '''
    f = transition_model(qht, k, delta_t, jt)
    return ss.norm.rvs(f,sig_v) # this process internalized sig_v
# observation
def g_theta(qht,sig_w,qt):
    """
        qht: all possible \hat{x}_{t}
        qt: observed
    """
    return ss.norm(qht,sig_w).pdf(qt)
# discrete inverse transform sampling
def dits(pdf,x,num):
    '''
        give x ~ pdf
    return: index of x that are been sampled according to pdf
    '''
    # TODO: use binary search - more efficient
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
            for j in range(1,len(cdf)):
                if (u[i] <= cdf[j]) and (u[i] > cdf[j-1]):
                    a[i] = j
    return ind[a.astype(int)]
# calculate CI
def cal_CI(qh: List[float],P: List[float], alpha: float = 0.05):
    '''
        give qh ~ P at time i
        calculate (1 - alpha)% CI
    return: lower bound, upper bound, MLE
    '''
    L = []
    U = []
    MLE = []
    for i in range(T):
        X_ind = np.argsort(qh[i]) # sort states
        pmf = P[i][X_ind] # calculate pmf
        cmf = pmf.cumsum() # cmf
        i_mle = np.argmax(pmf)
        i_lower = np.argmin(abs(cmf - alpha/2))
        i_upper = np.argmin(abs(cmf - (1-alpha/2)))
        MLE.append(qh[i][X_ind[i_mle]])
        L.append(qh[i][X_ind[i_lower]])
        U.append(qh[i][X_ind[i_upper]])
    return L, U, MLE
# running the actual model
def run_sMC(J: List[float], Q: List[float], k: float, delta_t: float,M:int, sig_v: float, sig_w: float):
    '''
        definitions same as the wrapper
    return: qh  - estiamted state in particles
            P   - weight associated with each particles
            A   - ancestor lineage
            M   - total number of particles
    '''
    # initialization---------------------------------
    # assume X0 = 0 --> Dirac Delta distribution at 0
    A = [np.arange(M)] # ancestor storage
    Qh = [np.zeros(M)] # state storage, Q_hat
    P = [np.ones(M)/M] # initial weight on particles are all equal

    # state estimation-------------------------------
    for t in range(T):
        # sampling from ancestor based on previous weight
        aa = dits(P[-1],A[-1], num = len(P[-1]))
        A.append(aa)
        # draw new state samples and associated weights based on last ancestor
        qht = Qh[-1][aa]
        pht = P[-1][aa]
        # compute new state and weights based on the model
        qhtp1 = f_theta(qht,k,delta_t,J[t],sig_v) 
        phtp1 = pht * g_theta(qhtp1, sig_w, Q[t])
        phtp1 = phtp1/phtp1.sum() # normalize
        # update the info
        Qh.append(qhtp1)
        P.append(phtp1)
    # remove auxillary parts
    qh = np.array(Qh)[1:]
    P = np.array(P)[1:]
    A = np.array(A)[1:]
    return qh, P, A


def run_pMCMC(k: float, sig_v: float,sig_w: float, qh: List[float], P: List[float], A: List[float], J: List[float] , Q: List[float], k: float, delta_t: float):
    '''
    pMCMC inside loop, run this entire function as many as possible
        theta           - let it be k, sig_v, sig_w for now
        nstep           - number of steps in MCMC
        qh              - estiamted state in particles
        P               - weight associated with each particles
        A               - ancestor lineage
        J,Q,k,delta_t   - same as defined above
    '''
    # sample an ancestral path based on final weight
    T = len(Q)
    M = qh.shape[1]
    B = np.zeros(T).astype(int)
    B[-1] = dits(P[-1],qh[-1],num = 1)
    for i in reversed(range(1,T)):
        B[i-1] =  A[i][B[i]]
    # B is the current lineage
    notB = np.arange(0,M)!=B[0]
    # draw new state samples and associated weights based on last ancestor
    qhu = qh[0]
    pht = P[0]
    # update the states based on the model
    qhu[notB] = f_theta(qh[0][notB],k,delta_t,J[0],sig_v) 
    phu = pht * g_theta(qhu, sig_w, Q[0])
    phu = phu/phu.sum() # normalize
    # update the info
    qh[0] = qhu
    P[0] = phu
    # state estimation-------------------------------
    for t in range(1,T):
        notB = np.arange(0,M)!=B[t]
        # sampling from ancestor based on previous weight
        aa = dits(P[t-1],A[t-1], num = len(notB)-1)
        A[t][notB] = aa
        # draw new state samples and associated weights based on last ancestor
        qht = qh[t][aa]
        pht = P[t-1][aa]
        # compute new state and weights based on the model
        qhtp1 = f_theta(qht,k,delta_t,J[t],sig_v)
        phtp1 = P[t] 
        phtp1[notB] = pht * g_theta(qhtp1, sig_w, Q[t])
        phtp1 = phtp1/phtp1.sum() # normalize
        # update the info
        qh[t][notB] = qhtp1
        P[t] = phtp1
    return qh, P, A
# TODO: Gibbs Sampler
def run_pGS(J,Q,k0, sig_v0, sig_w0, delta_t):
    # for the inital guess on theta_0 = {k_0, sig_v0, sig_w0}
    # run sMC to get a first step estimation
    qh, P, A = run_sMC(J, Q, k0, delta_t,M, sig_v0, sig_w0)
    # now update our knowledge on theta
    
    # and now run another sMC

    # and run another theta estimation
     

# %%
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
    # run_particle_filter_wrapper(J: List[float], Q: List[float], k: float,\
    # delta_t:float, sig_q: float, sig_v: float, sig_w: float, \
    # M: int = 100, plot = True)

    # read data
    precip = pd.read_csv("precip.csv", index_col = 0)
    precip = precip['45'].values
    # define constant
    l = 100 # control the length of the time series
    delta_t = 1./24/60*15
    k = 1
    sig_v = 0.1
    sig_w = 0.01 
    M = 100
    plot = True
    #
    J,Q = data_generation(precip, delta_t, k , l)
    Q_true = Q.copy()
    # pretend we know obs error
    Q += np.random.normal(0, sig_w,size = len(Q))
    T = len(Q) # total age
    k /= delta_t # adjust k
    
    qh, P, A = run_sMC(J, Q, k, delta_t,M,sig_v,sig_w)
    L, U, MLE = cal_CI(qh,P)

    # ------------
    if plot == True:
        plt.figure()

        plt.plot(np.linspace(0,T,len(Q)),Q,".", label = "Observation")
        plt.plot(np.linspace(0,T,len(Q_true)),Q_true,'k', label = "True")
        plt.plot(np.linspace(0,T,len(MLE)),MLE,'r:', label = "MLE")
        plt.fill_between(np.linspace(0,T,len(L)), L, U, color='b', alpha=.3, label = "95% CI")
        
        plt.legend(ncol = 4)
        plt.title(f"sig_v {sig_v}, sig_w {sig_w}")
        plt.xlim([600,630])

    qh, P, A = run_pMCMC(1.,qh, P, A, J, Q, k, delta_t)
    L, U, MLE = cal_CI(qh,P)
    # ------------
    if plot == True:
        plt.figure()

        plt.plot(np.linspace(0,T,len(Q)),Q,".", label = "Observation")
        plt.plot(np.linspace(0,T,len(Q_true)),Q_true,'k', label = "True")
        plt.plot(np.linspace(0,T,len(MLE)),MLE,'r:', label = "MLE")
        plt.fill_between(np.linspace(0,T,len(L)), L, U, color='b', alpha=.3, label = "95% CI")
        
        plt.legend(ncol = 4)
        plt.title(f"sig_v {sig_v}, sig_w {sig_w}")
        plt.xlim([600,630])

        

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
# %%
