# %%
# Cleaned up version of Nish's code
# Author: Esther Xu Fei
import pandas as pd
import matplotlib.pyplot as plt
from mesas.sas.model import Model
import seaborn as sns
import scipy.stats as ss
import numpy as np

# %%
def set_sas_specs(loc, mean, a_1, b, a_2, a_3, a_4, C_old):

    # find scales for each beta distribution
    scale_T1 = mean*((a_1+b)/a_1)
    scale_T2 = mean*((a_2+b)/a_2)
    scale_T3 = mean*((a_3+b)/a_3)
    scale_T4 = mean*((a_4+b)/a_4)

    # set sas_specs for each sas function
    sas_specs = {
    "T (mL)":{
        "T_SAS_function1": {
            "func": "beta",
                "args": {
                    "loc": loc,
                    "scale": scale_T1,
                    "a": a_1,
                    "b": b}},  
            
        "T_SAS_function2": {
            "func": "beta",
            "args": {
                "loc": loc,
                "scale": scale_T2,
                "a": a_2,
                "b": b}},
        "T_SAS_function3": {
            "func": "beta",
            "args": {
                "loc": loc,
                "scale": scale_T3,
                "a": a_3,
                "b": b
                }},
        "T_SAS_function4": {
            "func": "beta",
            "args": {
                "loc": loc,
                "scale": scale_T4,
                "a": a_4,
                "b": b}}
            
            
            }
        }
    
    solute_parameters = {
        "2H_P (permille)":{
            "observations": "T_C_obs (permille)",
            "C_old": C_old
            }
        
        }
    
        
    options = {
            "verbose" : False,
            "record_state": True
        }
    return sas_specs, solute_parameters, options, scale_T1, scale_T2, scale_T3, scale_T4

def run_sas_model(sas_specs, solute_parameters, options, df):
    model = Model(data_df = df, sas_specs = sas_specs, solute_parameters = solute_parameters, **options)
    model.run()
    results = model.data_df
    return results

def get_four_sas_functions(a_1, a_2, a_3, a_4, b, loc, scale_T1, scale_T2, scale_T3, scale_T4, num_samples=1000):
                           
    x_1 = np.linspace(ss.beta.ppf(0., a_1, b), ss.beta.ppf(1., a_1, b), num_samples)
    x_2 = np.linspace(ss.beta.ppf(0., a_2, b), ss.beta.ppf(1., a_2, b), num_samples)
    x_3 = np.linspace(ss.beta.ppf(0., a_3, b), ss.beta.ppf(1., a_3, b), num_samples)
    x_4 = np.linspace(ss.beta.ppf(0., a_4, b), ss.beta.ppf(1., a_4, b), num_samples)


    ST_1 = loc + scale_T1*x_1
    ST_2 = loc + scale_T2*x_2
    ST_3 = loc +scale_T3*x_3
    ST_4 = loc + scale_T4*x_4

    SAS_1 = ss.beta.pdf(x_1, a_1, b)
    SAS_2 = ss.beta.pdf(x_2, a_2, b)
    SAS_3 = ss.beta.pdf(x_3, a_3, b)
    SAS_4 = ss.beta.pdf(x_4, a_4, b)

    # add a fake point infront of the first point to make the plot look nicer
    ST_1 = np.insert(ST_1, 0, 0)
    ST_2 = np.insert(ST_2, 0, 0)
    ST_3 = np.insert(ST_3, 0, 0)
    ST_4 = np.insert(ST_4, 0, 0)
    SAS_1 = np.insert(SAS_1, 0, 0)
    SAS_2 = np.insert(SAS_2, 0, 0)
    SAS_3 = np.insert(SAS_3, 0, 0)
    SAS_4 = np.insert(SAS_4, 0, 0)

    return ST_1, ST_2, ST_3, ST_4, SAS_1, SAS_2, SAS_3, SAS_4

# %%
# SETTING UP YOUR MODEL =========================

# This is your root directory
root = '/Users/esthersida/Downloads/'
# This is the name of your input file
file_name = 'PL4_EXP1_multicomponent.csv'
# your results will be saved in this format
save_name = f'{file_name[:-4]}_results.csv'
# These are your model parameters
loc= 42.6935
mean= 4221.87
a_1=  1.13866
b= 7.99982
a_2= 1.0139
a_3=  1.0637
a_4=   1.04445
C_old= 8.97
rmse = 17.54
species = 'PL4'

# read your input
df = pd.read_csv(root + file_name)
# set up your model using your parameters
sas_specs, solute_parameters, options, scale_T1, scale_T2, scale_T3, scale_T4 = set_sas_specs(loc, mean, a_1, b, a_2, a_3, a_4, C_old)
# run your model
results = run_sas_model(sas_specs, solute_parameters, options, df)
# save your results so that you don't have to rerun the model every single time
results.to_csv(root + save_name)

# %%
# MAKING PLOTS =========================

# get your results, uncomment the following two lines if you have already run the model

# df = pd.read_csv(root + file_name)
# results = pd.read_csv(root + save_name)

obs = results['T_C_obs (permille)']
pred = results['2H_P (permille) --> T (mL)']

# get your four sas functions
ST_1, ST_2, ST_3, ST_4, SAS_1, SAS_2, SAS_3, SAS_4 = get_four_sas_functions(a_1, a_2, a_3, a_4, b, loc, scale_T1, scale_T2, scale_T3, scale_T4, num_samples=1000)


#plot the sim vs. obs timeseries

lw = 3.
lim = 670
tick_fontsize = 14
label_fontsize = 18
legend_fontsize = 14

fig, ax= plt.subplots(1,2, figsize=(14,6))
fig.suptitle(f'Spices: {species}', fontsize = 20)
ax[0].plot(df.index, obs, color = 'black', label = 'Observation')
ax[0].plot(df.index, pred, color = 'red', label = 'Simulation')
ax[0].set_xlabel('Time step (every 3 hr)', fontsize = label_fontsize)
ax[0].set_ylabel('2H (‰)', fontsize = label_fontsize)
ax[0].legend(frameon = False, fontsize = legend_fontsize, loc = 'upper left')
ax[0].tick_params(axis='x', labelsize=tick_fontsize)
ax[0].tick_params(axis='y', labelsize=tick_fontsize)
ax[0].set_title(f'RMSE = {rmse}', fontsize = label_fontsize)

ax[1].plot(ST_1, SAS_1, linewidth = lw, alpha = 0.75, linestyle = '-', label = '9 am - 12 pm')
ax[1].plot(ST_2, SAS_2, linewidth = lw, alpha = 0.75, linestyle = (10, (3,
1, 1, 1, 1, 1)), label = '12 - 3 pm')
ax[1].plot(ST_3, SAS_3, linewidth = lw, alpha = 0.75, linestyle = '-.', label = '3 - 6 pm')
ax[1].plot(ST_4, SAS_4, linewidth = lw, alpha = 0.75, linestyle = ':', label = '6 - 9 pm')
ax[1].axvline(12, color = 'black', linestyle = '--', label = 'Max plant storage')
ax[1].set_xlim([0,lim])
ax[1].set_ylabel('$ω(S_{T})$', fontsize = label_fontsize)
ax[1].set_xlabel('$S_{T}$ (mL)', fontsize = label_fontsize)
ax[1].legend(frameon = False, fontsize = legend_fontsize, loc = 'lower right')
ax[1].tick_params(axis='x', labelsize=tick_fontsize)
ax[1].tick_params(axis='y', labelsize=tick_fontsize)
ax[1].set_title(f'Loc: {loc}, Mean: {mean}, C_old: {C_old}', fontsize = label_fontsize)

fig.tight_layout()
fig.savefig(root + f'{species}_rmse_{rmse}.pdf')
# %%
