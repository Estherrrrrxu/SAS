# %%
import os

current_path = os.getcwd()

if current_path[-11:] != "tests_mesas":
    os.chdir("tests_mesas")
    print("Current working directory changed to 'tests_mesas'.")
import sys

sys.path.append("../")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


# %%
# SET FOLDERS
# ================================================================
data_root = "/Users/esthersida/pMESAS/mesas"
pre_processed_data = f"{data_root}/data_preprocessed.csv"
raw_data = f"{data_root}/data.csv"

if not os.path.exists(pre_processed_data):
    # Data from publicated paper
    df = pd.read_csv(raw_data, index_col=1, parse_dates=True)
    df.columns = ["timestep", "J", "C in", "Q", "ET", "C out", "S_scale"]
    time = df.index

    # Orginal data only used for tracking whether measurements being made
    check_file = pd.read_csv(
        f"{data_root}/weekly_rainfall.csv", index_col=5, parse_dates=True
    )

    # Find where C in is observed
    df["is_obs_input"] = check_file["Cl mg/l"]
    df["is_obs_input"] = df["is_obs_input"].notna()
    df["is_obs_input_filled"] = False

    # Replace randomly generated cover data
    df["C in raw"] = df["C in"][df["is_obs_input"]]
    df["C in raw"] = df["C in raw"].backfill()
    df["C in raw"][df["J"] == 0.0] = 0.0

    # Replace randomly sampled data with monthly mean
    ill_vals = df[["C in", "J"]][df["C in"] != df["C in raw"]]
    monthly_means = ill_vals["C in"].groupby([ill_vals.index.month]).mean()
    ill_vals["C in"] = monthly_means[ill_vals.index.month].to_list()

    plt.figure(figsize=(6, 2))
    plt.plot(monthly_means, "*:")
    plt.xlim([0.8, 12.2])
    plt.xlabel("Month")
    plt.ylabel("Concentration [mg/L]")
    plt.title("Monthly mean concentration used to fill missing data")
    plt.tight_layout()
    plt.savefig(f"{data_root}/monthly_mean_concentration_filler.pdf")
    #

    # define continuous time interval, mark the end of continuous filled series as observation
    fake_obs = (ill_vals.index.to_series().diff().dt.days >= 6).to_list()[1:]
    fake_obs.append(True)
    ill_vals["is_obs"] = fake_obs

    # deal with across month condition
    ill_vals["group"] = ill_vals["is_obs"].cumsum()
    ill_vals["group"][ill_vals["is_obs"] == True] -= 1
    for i in range(ill_vals["group"].iloc[-1] + 1):
        if len((ill_vals["C in"][ill_vals["group"] == i]).unique()) > 1:
            ill_vals["C in"][ill_vals["group"] == i] = (
                ill_vals["C in"][ill_vals["group"] == i]
                * ill_vals["J"][ill_vals["group"] == i]
            ).sum() / ill_vals["J"][ill_vals["group"] == i].sum()

    # Combine the filled data with the original data
    df["C in raw"][df["C in"] != df["C in raw"]] = ill_vals["C in"]
    df["is_obs_input"][df["C in"] != df["C in raw"]] = ill_vals["is_obs"]
    df["is_obs_input_filled"][df["C in"] != df["C in raw"]] = True

    df["is_obs_ouput"] = df["C out"].notna()

    # rename the columns
    df.columns = [
        "timestep",
        "J",
        "C in fake",
        "Q",
        "ET",
        "C out",
        "S_scale",
        "is_obs_input",
        "is_obs_input_filled",
        "C in",
        "is_obs_output",
    ]
    df.to_csv(pre_processed_data)
else:
    df = pd.read_csv(pre_processed_data, index_col=0, parse_dates=True)
    # VISUALIZATION
    start_ind, end_ind = 2250, 3380
    data = df.iloc[start_ind:end_ind]
    # reset df index
    # data = data.reset_index(drop=True)
    fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    # precipitation
    J = ax[0].bar(data.index, data["J"], label="J")

    ax0p = ax[0].twinx()
    CJ = ax0p.scatter(
        data.index[data["C in"] > 0],
        data["C in"][data["C in"] > 0],
        color="r",
        marker=".",
        label=r"$Observed\ C_J$",
    )
    temp_ind = np.logical_and(data["is_obs_input_filled"].values, data["C in"].values > 0)
    CJ1 = ax0p.scatter(
        data.index[temp_ind],
        data["C in"][temp_ind],
        color="k",
        marker=".",
        label=r"$Filled\ C_J$",
    )


    Q = ax[1].plot(data.index, data["Q"], label="Q")

    ax1p = ax[1].twinx()
    CQ = ax1p.scatter(
        data.index[data["C out"] > 0],
        data["C out"][data["C out"] > 0],
        color="r",
        marker=".",
        label=r"$C_Q$",
    )

    # evapotranspiration
    ax[2].plot(data["ET"])

    # storage
    ax[3].plot(data["S_scale"])

    # settings
    ax[0].set_title("Input - Precipitation", fontsize=16)
    ax[1].set_title("Output 1 - Discharge", fontsize=16)
    ax[2].set_title("Output 2 - Evapotranspiration (ET)", fontsize=16)
    ax[3].set_title("Maximum storage", fontsize=16)

    ax[0].set_ylabel("Precipitation [mm/d]", fontsize=14)
    ax0p.set_ylabel("Concentration [mg/L]", fontsize=14)
    ax[0].set_ylim([data["J"].max() * 1.02, data["J"].min()])
    ax0p.set_ylim([0.0, data["C in"].max() * 1.2])

    ax[1].set_ylabel("Discharge [mm/d]", fontsize=14)
    ax1p.set_ylabel("Concentration [mg/L]", fontsize=14)

    ax[2].set_ylabel("ET [mm/h]", fontsize=14)
    ax[3].set_ylabel("S max [mm]", fontsize=14)


    lines = [J, CJ, CJ1]
    labels = [line.get_label() for line in lines]
    ax[0].legend(lines, labels, loc="upper right", fontsize=14, ncol=3)

    lines = [Q[0], CQ]
    labels = [line.get_label() for line in lines]
    ax[1].legend(lines, labels, loc="upper right", fontsize=14, ncol=2)

    ax[3].set_xlim([data.index[0], data.index[-1]])
    fig.tight_layout()

    plt.savefig(f"{data_root}/data_visualization.pdf")

    # %%
    # show right skewness of the data
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    d = df[df["C in"] > 0]
    d = d[d['is_obs_input']]
    d = d[d['is_obs_input_filled'] == False]
    d = d["C in"]    
    shape, loc, scale = ss.lognorm.fit(d, floc=0)

    ax[0].hist(d, bins=100, label=r"$C_J$",density=True)
    ax[0].plot(np.linspace(min(d), max(d), 1000), ss.lognorm.pdf(np.linspace(min(d), max(d), 1000), shape, loc, scale), label = "Fitted")
    ax[0].set_xlabel("Concentration [mg/L]")
    ax[0].set_ylabel("Density")
    ax[0].set_title("Original data")
    ax[0].legend(fontsize=12, loc="upper right", frameon=False)
    ax[0].set_xlim([min(d), max(d)])

    ax[1].hist(np.log(d), bins=100, label=r"$log(C_J)$", density=True)
    ax[1].plot(np.linspace(np.log(min(d)), np.log(max(d)), 1000), ss.norm.pdf(np.linspace(np.log(min(d)), np.log(max(d)), 1000), np.log(scale), shape), label = "Converted")
    ax[1].set_xlabel("log(Concentration)")
    ax[1].set_title("Log-transformed data")
    ax[1].legend(fontsize=12, loc="upper right", frameon=False)
    ax[1].set_xlim([np.log(min(d)),  np.log(max(d))])
    

    fig.tight_layout()
    plt.savefig(f"{data_root}/observed_concentration_histogram.pdf")
    
# %%
