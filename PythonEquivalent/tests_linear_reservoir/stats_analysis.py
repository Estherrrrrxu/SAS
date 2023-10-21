# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
result_root = "/Users/esthersida/pMESAS/Results/TestLR/WhiteNoise"
case_name = "Almost perfect data"
df = pd.read_csv(f"{result_root}/RMSE_{case_name}.csv", index_col=0)
df.columns = [
    "Input RMSE",
    "Output RMSE",
    "Signal to Noise",
    "True k",
    "Input st.dev",
    "threshold",
]
df["Input theoretical RMSE"] = df["Input st.dev"]/df["Signal to Noise"]
df["Output theoretical RMSE"] = np.sqrt(df["Input theoretical RMSE"]**2 / (1 - (1 - df["True k"])**2))
# %%
df_subset = df[df["threshold"] == 20]
df_subset = df_subset[df_subset["Signal to Noise"] == 1.]

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
sns.barplot(
    x="True k",
    y=df_subset["Input RMSE"] - df_subset["Input theoretical RMSE"],
    hue="Input st.dev",
    data=df_subset,
    ax=ax[0],
    legend=False,
    palette="muted",
)
sns.barplot(
    x="True k",
    y=df_subset["Output RMSE"] - df_subset["Output theoretical RMSE"],
    hue="Input st.dev",
    data=df_subset,
    ax=ax[1],
    palette="muted",
)

ax[0].set_xlabel("")
ax[1].set_xlabel("True k", fontsize=15)
ax[0].set_ylabel("")
ax[1].set_ylabel("")
ax[0].set_title("Input RMSE - theoretical input RMSE", fontsize = 15)
ax[1].set_title("Output RMSE - theoretical output RMSE", fontsize = 15)
ax[1].legend(frameon=False, title="Input st.dev", fontsize=12, loc = "lower right")
plt.rcParams['legend.title_fontsize'] = 'larger'
fig.tight_layout()
# %%
df_subset = df[df["threshold"] == 20]
df_subset = df_subset[df_subset["Input st.dev"] == 1.]

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
sns.barplot(
    x="True k",
    y=df_subset["Input RMSE"] - df_subset["Input theoretical RMSE"],
    hue="Signal to Noise",
    data=df_subset,
    ax=ax[0],
    legend=False,
    palette="muted",
)
sns.barplot(
    x="True k",
    y=df_subset["Output RMSE"] - df_subset["Output theoretical RMSE"],
    hue="Signal to Noise",
    data=df_subset,
    ax=ax[1],
    palette="muted",
)

ax[0].set_xlabel("")
ax[1].set_xlabel("True k", fontsize=15)
ax[0].set_ylabel("")
ax[1].set_ylabel("")
ax[0].set_title("Input RMSE - theoretical input RMSE", fontsize = 15)
ax[1].set_title("Output RMSE - theoretical output RMSE", fontsize = 15)
ax[1].legend(frameon=False, title="Signal to Noise", fontsize=12, loc = "lower right")
plt.rcParams['legend.title_fontsize'] = 'larger'
fig.tight_layout()
# %%
df_subset = df[df["Signal to Noise"] == 1.]
df_subset = df_subset[df_subset["Input st.dev"] == 1.]

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
sns.barplot(
    x="True k",
    y=df_subset["Input RMSE"] - df_subset["Input theoretical RMSE"],
    hue="threshold",
    data=df_subset,
    ax=ax[0],
    legend=False,
    palette="muted",
)
sns.barplot(
    x="True k",
    y=df_subset["Output RMSE"] - df_subset["Output theoretical RMSE"],
    hue="threshold",
    data=df_subset,
    ax=ax[1],
    palette="muted",
)

ax[0].set_xlabel("")
ax[1].set_xlabel("True k", fontsize=15)
ax[0].set_ylabel("")
ax[1].set_ylabel("")
ax[0].set_title("Input RMSE - theoretical input RMSE", fontsize = 15)
ax[1].set_title("Output RMSE - theoretical output RMSE", fontsize = 15)
ax[1].legend(frameon=False, title="Threshold", fontsize=12, loc = "lower right")
plt.rcParams['legend.title_fontsize'] = 'larger'
fig.tight_layout()

# %%
