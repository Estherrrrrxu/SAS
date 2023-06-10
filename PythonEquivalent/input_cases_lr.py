# %%
import pandas as pd
import numpy as np
from functions.utils import plot_MLE, plot_scenarios, plot_base
import matplotlib.pyplot as plt
from dataclasses import dataclass

# %%
df = pd.read_csv("Data/linear_reservoir.csv", index_col= 0)
# st, et = 20, 100
st, et = 300, 400


original = df[st:et]
original.index = range(len(original))

instant_gaps_2_d = original[::2]
instant_gaps_2_d.index = range(len(instant_gaps_2_d))

instant_gaps_5_d = original[::5]
instant_gaps_5_d.index = range(len(instant_gaps_5_d))

weekly_bulk = original.groupby(original.index//7).mean()
df_temp = pd.merge(original, weekly_bulk, on='index', how='left')
df_temp = df_temp.drop(df_temp.columns[1:5], axis=1)
df_temp.columns = original.columns
df_temp.iloc[-1,1:] = weekly_bulk.iloc[-1,1:]
ind_w = df_temp['Q_obs'].notna()
weekly_bulk = df_temp.fillna(method='bfill')
weekly_bulk['is obs'] = ind_w


biweekly_bulk = original.groupby(original.index//14).mean()
biweekly_bulk['index'] = (biweekly_bulk['index']+0.5).astype(int)
ind = biweekly_bulk['index'].values
df_temp = pd.merge(original, biweekly_bulk, on='index', how='left')
df_temp = df_temp.drop(df_temp.columns[1:5], axis=1)
df_temp.columns = original.columns
df_temp.iloc[-1,1:] = biweekly_bulk.iloc[-1,1:]
biweekly_bulk = df_temp.fillna(method='bfill')
biweekly_bulk['is obs']=biweekly_bulk['index'].isin(ind)

weekly_bulk_true_q = weekly_bulk.copy()
weekly_bulk_true_q["Q_obs"][original.index%7!=0] = np.nan
weekly_bulk_true_q["Q_obs"][original.index%7==0] = original["Q_obs"][original.index%7==0].values
weekly_bulk_true_q["is obs"] = ind_w

# %%
plot_base(original, original)
plot_base(original, instant_gaps_2_d)
plot_base(original, instant_gaps_5_d)
plot_base(original, weekly_bulk)
plot_base(original, biweekly_bulk)
plot_base(original, weekly_bulk_true_q)
# %%
@dataclass
class Cases:
    df: pd.DataFrame
    df_obs: pd.DataFrame
    interval: int
    case_name: str

perfect = Cases(df=original, df_obs=original, interval=1, case_name="Almost perfect data")

instant_gaps_2_d = Cases(df=original, df_obs=instant_gaps_2_d, interval=2, case_name="Instant measurement w/ gaps of 2 days")

instant_gaps_5_d = Cases(df=original, df_obs=instant_gaps_5_d, interval=5, case_name="Instant measurement w/ gaps of 5 days")

weekly_bulk = Cases(df=original, df_obs=weekly_bulk, interval=1, case_name="Weekly bulk")
biweekly_bulk = Cases(df=original, df_obs=biweekly_bulk, interval=1, case_name="Biweekly bulk")
weekly_bulk_true_q = Cases(df=original, df_obs=weekly_bulk_true_q, interval=1, case_name="Weekly bulk w/ true Q")
# %%
