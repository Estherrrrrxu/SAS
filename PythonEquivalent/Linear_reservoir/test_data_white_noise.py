# %%
import pandas as pd
import numpy as np
import sys
sys.path.append('../') 
from functions.utils import plot_base, create_bulk_sample, plot_bulk
from dataclasses import dataclass
from typing import Optional, List

# %%
# Read original file and get a part of it
df = pd.read_csv("../Data/WN_ipt_0.002_obs_6e-05.csv", index_col= 0)
# st, et = 20, 100
st, et = 0,len(df)
plot = True
# %%
# For instantaneously observed data
# observation made at each time step
original = df[st:et]
original['index'] = range(len(original))
# observation made at each 2 time steps
instant_gaps_2_d = original[::2]
instant_gaps_2_d.index = range(len(instant_gaps_2_d))

# observation made at each 5 time steps
instant_gaps_5_d = original[::5]
instant_gaps_5_d.index = range(len(instant_gaps_5_d))

if plot:
    plot_base(original, original)
    plot_base(original, instant_gaps_2_d)
    plot_base(original, instant_gaps_5_d)

# %%
# For bulk observed data
weekly_bulk = create_bulk_sample(original, 7)
biweekly_bulk = create_bulk_sample(original, 14)

weekly_bulk_true_q = weekly_bulk.copy()
ind = weekly_bulk_true_q['is_obs']
weekly_bulk_true_q["Q_obs"][ind==False] = np.nan
weekly_bulk_true_q["Q_obs"][ind==True] = weekly_bulk_true_q["Q_true"][ind==True]
weekly_bulk_true_q = weekly_bulk_true_q.fillna(method='bfill')

if plot:
    plot_bulk(original, weekly_bulk)
    plot_bulk(original, biweekly_bulk)
    plot_bulk(original, weekly_bulk_true_q)

# %%
original['is_obs'] = True
instant_gaps_2_d['is_obs'] = True
instant_gaps_5_d['is_obs'] = True
# %%
@dataclass
class Cases:
    df: pd.DataFrame
    df_obs: pd.DataFrame
    case_name: str
    obs_made: Optional[bool or int or List[bool]] = True

perfect = Cases(
    df=original, 
    df_obs=original, 
    obs_made=1, 
    case_name="Almost perfect data"
    )

instant_gaps_2_d = Cases(
    df=original, 
    df_obs=instant_gaps_2_d, 
    obs_made=2, 
    case_name="Instant measurement w/ gaps of 2 days"
    )

instant_gaps_5_d = Cases(
    df=original, 
    df_obs=instant_gaps_5_d, 
    obs_made=5, 
    case_name="Instant measurement w/ gaps of 5 days"
    )

weekly_bulk = Cases(
    df=original, 
    df_obs=weekly_bulk, 
    obs_made=weekly_bulk["is_obs"].values, 
    case_name="Weekly bulk"
    )
biweekly_bulk = Cases(
    df=original, 
    df_obs=biweekly_bulk, 
    obs_made=biweekly_bulk["is_obs"].values, 
    case_name="Biweekly bulk"
    )
weekly_bulk_true_q = Cases(
    df=original, 
    df_obs=weekly_bulk_true_q, 
    obs_made=weekly_bulk_true_q["is_obs"].values, 
    case_name="Weekly bulk w/ true Q"
    )
# %%
