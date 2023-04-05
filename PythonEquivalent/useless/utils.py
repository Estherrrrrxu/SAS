# %%
import numpy as np
import matplotlib.pyplot as plt

# %%



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
def plot_MLE(state,df,left = 0, right = 500):
    X = state.X
    A = state.A
    W = state.W
    R = state.R
    J_obs = df['J_obs'].values
    Q_obs = df['Q_obs'].values
    K = len(J_obs)

    B = np.zeros(K+1).astype(int)
    B[-1] = _inverse_pmf(W,A[:,-1], num = 1)
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
    # plt.title(f"sig_v {round(sig_v,5)}, sig_w {sig_w}")
    plt.xlim([left,right])
    plt.tight_layout()
    return MLE