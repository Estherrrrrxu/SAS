# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from functions.models import transition_model
import scipy.stats as ss
# %%
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