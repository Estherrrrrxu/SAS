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
        plot: Optional[bool] = False,
        observation_mode: Optional[str] = "perfect",
) -> None:

    # %%
    # For instantaneously observed data
    
    # observation made at each time step
    st, et = interval[0], interval[1]
    original = df[st:et]
    original['index'] = range(len(original))
    original['is_obs'] = True

    if observation_mode == "perfect":

        if plot:
            plot_base(original)

        perfect = Cases( 
        df_obs=original, 
        obs_made=1, 
        case_name="Almost perfect data"
        )

        return perfect
    elif observation_mode == "deci_2d":

        # observation made at every 2 time steps
        deci_2d = original.copy()
        deci_2d["is_obs"] = False
        deci_2d["is_obs"][::2] = True
        
        if plot:
            plot_base(deci_2d)
        
        deci_2d = Cases(
            df_obs=deci_2d, 
            obs_made=2, 
            case_name="Decimated every 2d"
            )

        return deci_2d
    
    elif observation_mode == "deci_4d":

        # observation made at every 5 time steps
        deci_4d = original.copy()
        deci_4d["is_obs"] = False
        deci_4d["is_obs"][::4] = True

        if plot:
            plot_base(deci_4d)
        
        deci_4d = Cases(
            df_obs=deci_4d, 
            obs_made=4, 
            case_name="Decimated every 4d"
            )
        
        return deci_4d
    
    elif observation_mode == "deci_7d":

        # observation made at every 7 time steps
        deci_7d = original.copy()
        deci_7d["is_obs"] = False
        deci_7d["is_obs"][::7] = True

        if plot:
            plot_base(deci_7d)

        deci_7d = Cases(
            df_obs=deci_7d, 
            obs_made=7, 
            case_name="Decimated every 7d"
            )
        
        return deci_7d
    
    elif observation_mode == "bulk_2d":

        bulk_2d = create_bulk_sample(original, 2)

        if plot:
            plot_bulk(bulk_2d)

        bulk_2d = Cases(
            df_obs=bulk_2d, 
            obs_made=2, 
            case_name="Bulk every 2d"
            )
        
        return bulk_2d
    
    elif observation_mode == "bulk_4d":

        bulk_4d = create_bulk_sample(original, 4)

        if plot:
            plot_bulk(bulk_4d)
        
        bulk_4d = Cases(
            df_obs=bulk_4d, 
            obs_made=4, 
            case_name="Bulk every 4d"
            )

        return bulk_4d
    
    elif observation_mode == "bulk_7d":

        bulk_7d = create_bulk_sample(original, 7)

        if plot:
            plot_bulk(bulk_7d)
        
        bulk_7d = Cases(
            df_obs=bulk_7d, 
            obs_made=7, 
            case_name="Bulk every 7d"
            )

        return bulk_7d
    
    else:
        raise ValueError("Invalid observation mode")









    


# %%
