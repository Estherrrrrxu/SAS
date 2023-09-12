# %%
import pandas as pd
import sys
sys.path.append("../")
from functions.get_dataset import get_different_input_scenarios


# %%
df = pd.read_csv("../Data/WhiteNoise/stn_1_2.csv", index_col= 0)
interval = [0,100]

perfect, instant_gaps_2_d, instant_gaps_5_d, weekly_bulk, biweekly_bulk, weekly_bulk_true_q = get_different_input_scenarios(df, interval, plot=True)
# %%
