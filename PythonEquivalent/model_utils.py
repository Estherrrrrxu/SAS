# %%
import numpy as np

# %%
# "resampling.m" - resampling process
def resampling(q):
    """
        q - 1 x N array
    """
    # gives an array q
    N = len(q)
    # generate (1, N) array in [0,1]
    u = np.random.uniform(0,1,N)
    # cdf for q 
    qc = q.cumsum()
    qc = qc/qc[-1]
    # indices for sorted rv + cdf(q)
    ind_temp = np.argsddort(np.append(u,qc))
    # return a index
    return np.where(ind_temp< N)[0] - (np.linspace(0,N,N)).astype(int)


# %%
# "fnObs_nl.m" - the observation function
def fnObs_nl(x):
    """
        x - the hidden state
    """
    f  = 1*x # TODO: here substitute w/ MESAS obs 
    return f

# %%
# 'pgas_MLE.m' - the body of PGAS algorithm, SIR in SMC

# return the sample paths of x_{1:T}
# X -- the Markov Chain of x_{1:T}, size = [numMCMC, T]

def pgas_parBayes(numMCMC, obs, Np, ssmPar):
    T           = len(obs) # total time steps
    X           = np.zeros(numMCMC, T) # stored states at each chain, each timestep
    len_param   = len(ssmPar.coefs)
    theta       = np.zeros(numMCMC, len_param)
    svar        = np.zeros(numMCMC, 1) # ? 
    ess         = np.aeros(numMCMC, T) # ?

    theta0      = np.random.randn()
