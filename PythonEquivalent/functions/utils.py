# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
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
# =================
# Plotting section
# =================
def plot_input(df,J_obs, Q_obs):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(df['J_true'],label = "J true")
    plt.plot(J_obs,"*",label = "J obs")
    plt.legend()
    plt.title('J')
    plt.subplot(2,1,2)
    plt.plot(df['Q_true'],label = "Q true")
    plt.plot(Q_obs,"*",label = "Q obs")
    plt.title('Q')
    plt.legend()
    plt.tight_layout()
    return
# plotting functions
def plot_MLE(X,A,W,R,K,df,J_obs, Q_obs,sig_v,sig_w,left = 0, right = 500):
    B = np.zeros(K+1).astype(int)
    B[-1] = dits(W,A[:,-1], num = 1)
    for i in reversed(range(1,K+1)):
        B[i-1] =  A[:,i][B[i]]
    MLE = np.zeros(K+1)
    MLE_R = np.zeros(K)
    for i in range(K+1):
        MLE[i] = X[B[i],i]
    for i in range(K):
        MLE_R[i] = R[B[i+1],i]   
    T = len(X[0])-1
    # ------------
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(df['J_true'],label = "J true")
    plt.plot(J_obs,"*",label = "J obs")
    plt.plot(np.linspace(0,T,len(MLE_R)),MLE_R,'r:', label = "One Traj/MLE")
    plt.legend(frameon = False)
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
    return