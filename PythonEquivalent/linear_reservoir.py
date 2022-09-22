# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
# %%
"""
    Considering the following condition:
    Q = theta * S
    J - Q = dS/dT
    Goal: use particle filter to estimate theta
"""
l = 100 # control the length of the time series
delta_t = 1./24/60*15
k = 0.01/delta_t
Q_init = 0.
# get precipitation data
precip = pd.read_csv("precip.csv", index_col = 0)
length = round(len(precip)/l)

# for 
J = precip['45'].values[:length]
Q = np.zeros(length)
time = np.linspace(0, length*delta_t,length)
def model(qt,k,delta_t,jt):
        qtp1 = (1 - k * delta_t) * qt + k * delta_t * jt
        return qtp1
for i in range(length-1):
    #Q[i:] += J[i]*k*delta_t*np.exp(-k*delta_t*np.linspace(0,length*delta_t,length - i))
    Q[i+1] = model(Q[i], k, delta_t, J[i])



C_old = 1000
C_J = C_old + np.random.randn(length)*1000.0
C_Q = np.zeros(length)
kappa = k*delta_t
# j - timestep
# i - age step

# for all timesteps
for t in range(length):
    # now the maximum age in the system is t
    pq = np.zeros(t+2) # store cq that goes away            
    # when T = 0
    pq[0] = (kappa+np.exp(-kappa)-1)*J[t]/kappa/delta_t/Q[t]
    # get C_Q at T == 0
    C_Q[t] += C_J[t]*pq[0]*delta_t

    # when T > 0
    for T in range(1,t+1):
        # calculate SAS
        pq[T] = (1-np.exp(-kappa)*J[t-T])               
        # get C_Q
        C_Q[t] += C_J[t-T]*pq[T]*delta_t
    C_Q[t] += C_old*(1-pq.sum()*delta_t)
# %%
df = pd.DataFrame({"J":J,"C_J":C_J,"Q":Q,"C_Q":C_Q})
df = df.dropna()
# %%
plt.subplot(4,1,1)
plt.plot(df['J'])
plt.subplot(4,1,2)
plt.plot(df['C_J'])
plt.subplot(4,1,3)
plt.plot(df['Q'])
plt.subplot(4,1,4)
plt.plot(df['C_Q'])
plt.title("C_Q")
plt.tight_layout()
# df.to_csv("synthetic.csv")
# %%
          
def run_scenario(sig_q,sig_v, sig_w):
    # TODO: add prior on k and update k
    # initialization
    # data
    J = df['J'].values
    Q = df['Q'].values + np.random.normal(0,sig_w,size = len(df))
    Q_true = df['Q'].values
    T = len(df) # total age
    M = 100 # total number of particles
    # assume X0 = 0 --> Dirac Delta distribution at 0
    Q = np.append(0,Q) # q index is one more than t
    # inital weight = 1/M
    W = [np.ones(M)/M]
    # ancestor storage
    A = [np.arange(M)]
    Xh = [np.zeros(M)]
    P = [np.ones(M)/M]
    # initial guess on k
    # delta_t  = 1
    #k = k/100
    # set time diff
    
    # state transition
    def f_theta(qt,k,delta_t,jt,sig_v):
        #f = (1 - k * delta_t) * qtt + k * delta_t * jt
        f = model(qt, k, delta_t, jt)
        return ss.norm.rvs(f,sig_v)
    # observation
    def g_theta(qth,sig_w):
        return ss.norm(qth,sig_w)
    # importance sampling probability, q_hat ~ N(q0,Q.std())
    def q_theta(qt, std = sig_q):
        return ss.norm(qt, std)
    # discrete inverse transform sampling
    def dits(pdf,x):
        cdf = pdf.cumsum()
        u = np.random.uniform(size = len(x))
        a = np.zeros(len(x))
        for i in range(len(cdf)):
            if u[i] < cdf[0]:
                a[i] = 0
            else:
                for j in range(1,len(cdf)):
                    if (u[i] <= cdf[j]) and (u[i] > cdf[j-1]):
                        a[i] = j
        # TODO: use binary search
        # left, right = 0, len(cdf)
        # if u < cdf[0]:
        #     a = 0
        # elif u == cdf[-1]:
        #     a = len(cdf)
        # else:
        #     while left < right:
        return a.astype(int)
                

    #P = []
    for t in range(T):
        # find ancestor
        aa = dits(P[-1],A[-1])
        A.append(aa)
        # draw new state samples based on last obs
        #xx = q_theta(Xh[-1][aa]).rvs(M)
        xht = Xh[-1][aa]
        pht = P[-1][aa]
        # compute weights
        xhtp1 = f_theta(xht,k,delta_t,J[t],sig_v)#.pdf(xx) 
        phtp1 = pht * g_theta(xhtp1,sig_w).pdf(Q[t+1])
        #ww /= q_theta(Q[t+1]).pdf(xx)
        phtp1 = phtp1/phtp1.sum()
        #W.append(wtp1)           
        Xh.append(xhtp1)
        P.append(phtp1)
    # implicit sampling: random sampling --> make use of implicit sampling
    # TODO: implement 95% uncertainty bound
    xh = np.array(Xh)
    P = np.array(P)
    # plt.scatter(np.tile(np.linspace(0,T,T+1),(50,1)),xh[:,:50].T, alpha=W[:,:50])
    # plt.plot(Q,"k")
    xh = xh[1:]
    L = []
    H = []
    MLE = []
    for i in range(T):
        X_ind = np.argsort(xh[i])
        hist = P[i][X_ind]
        p = hist.cumsum()
        i_max = np.argmax(hist)
        i_05 = np.argmin(abs(p - 0.025))
        i_95 = np.argmin(abs(p - 0.975))
        MLE.append(xh[i][X_ind[i_max]])
        L.append(xh[i][X_ind[i_05]])
        H.append(xh[i][X_ind[i_95]])
        xh = np.array(Xh)
    plt.plot(np.linspace(0,T,len(MLE)),MLE,'r', label = "MLE")
    plt.plot(np.linspace(0,T,len(Q)),Q,".", label = "Observation")
    plt.plot(np.linspace(0,T,len(Q_true)),Q_true,'k', label = "True")

    plt.fill_between(np.linspace(0,T,len(L)), L, H, color='b', alpha=.3, label = "95% CI")
    plt.legend(ncol = 3)
    plt.title(f"sig_q {sig_q}, sig_v {sig_v}, sig_w {sig_w}")
    plt.xlim([0,100])
    plt.ylim([-0.001,0.007])
    return L, H, MLE, P, W, A


# %%
L,H,MLE, P, W, A = run_scenario(sig_q = 0.1,sig_v = 0.001, sig_w = 0.001)


# %%
#L,H,MLE = run_scenario(sig_q = 1,sig_v = 0.0001, sig_w = 0.01)

