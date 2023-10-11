# %%
import pandas as pd
import numpy as np

from functions.utils import plot_base, plot_bulk, create_bulk_sample
from dataclasses import dataclass
from typing import Optional, List
# %%
@dataclass
class Cases:
    df_obs: pd.DataFrame
    case_name: str
    obs_made: Optional[bool or int or List[bool]] = True
# %%
def get_different_input_scenarios(
        df: pd.DataFrame,
        interval: List[int],
        plot: Optional[bool] = False
) -> None:

    # %%
    # For instantaneously observed data
    # observation made at each time step
    st, et = interval[0], interval[1]
    original = df[st:et]
    original['index'] = range(len(original))
    original['is_obs'] = True
    # observation made at each 2 time steps
    instant_gaps_2_d = original.copy()
    instant_gaps_2_d["is_obs"] = False
    instant_gaps_2_d["is_obs"][::2] = True
    # observation made at each 5 time steps
    instant_gaps_5_d = original.copy()
    instant_gaps_5_d["is_obs"] = False
    instant_gaps_5_d["is_obs"][::5] = True

    if plot:
        plot_base(original, original)
        plot_base(original, instant_gaps_2_d)
        plot_base(original, instant_gaps_5_d)

    #
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

    
    perfect = Cases( 
        df_obs=original, 
        obs_made=1, 
        case_name="Almost perfect data"
        )

    instant_gaps_2_d = Cases(
        df_obs=instant_gaps_2_d, 
        obs_made=2, 
        case_name="Instant measurement w/ gaps of 2 days"
        )

    instant_gaps_5_d = Cases(
        df_obs=instant_gaps_5_d, 
        obs_made=5, 
        case_name="Instant measurement w/ gaps of 5 days"
        )

    weekly_bulk = Cases(
        df_obs=weekly_bulk, 
        obs_made=weekly_bulk["is_obs"].values, 
        case_name="Weekly bulk"
        )
    
    biweekly_bulk = Cases(
        df_obs=biweekly_bulk, 
        obs_made=biweekly_bulk["is_obs"].values, 
        case_name="Biweekly bulk"
        )
    
    weekly_bulk_true_q = Cases(
        df_obs=weekly_bulk_true_q, 
        obs_made=weekly_bulk_true_q["is_obs"].values, 
        case_name="Weekly bulk w/ true Q"
        )

    return perfect, instant_gaps_2_d, instant_gaps_5_d, weekly_bulk, biweekly_bulk, weekly_bulk_true_q

# %%
