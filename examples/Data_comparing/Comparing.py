# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 16:27:40 2026

@author: verwegen
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import glob
import os
# from brokenaxes import brokenaxes

#%% import observed


path = r"C:\Users\verwegen\Thesis\Data\Luvuvhu\Comparing\Observed_A9H012_old_daily.txt"

df_obs_old = pd.read_csv(
    path,
    sep=r'\s+',
    usecols=[0, 1], 
    parse_dates=True, 
    index_col=0)

df_obs_old = df_obs_old[df_obs_old["obs"] != 170]

df_obs_old_month = df_obs_old.resample("ME").mean()

#%% quick plot of the observed data
plt.figure(figsize=(10,6))
plt.plot(df_obs_old.index, df_obs_old["obs"], 
         label= 'Observed', 
        #  marker = 'o'
         )
plt.xlabel("baseline_time")
plt.ylabel("Discharge")
plt.title("Baseline discharge in current state vs observed data")
plt.legend()
# plt.xlim(pd.to_datetime("2001-01-01"),
#          pd.to_datetime("2002-01-01"))
plt.ylim(0,200)
plt.show()


#%% import WRC old method

path1 = r"C:\Users\verwegen\Thesis\Data\Luvuvhu\Comparing\WRC_A91H.ans"

# Adjusted columns to match your data: year, then Oct -> Sep
cols = [
    "year",
    "Oct", "Nov", "Dec",
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep",
    "annual"
]

df_wrc = pd.read_csv(
    path1,
    delim_whitespace=True,
    header=None,
    names=cols
)

# Melt with the same order
df_wrc_long = df_wrc.melt(
    id_vars="year",
    value_vars=[
        "Oct", "Nov", "Dec",
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep"
    ],
    var_name="month",
    value_name="value"
)

# Map months to numbers (calendar order)
month_map = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

df_wrc_long["month_num"] = df_wrc_long["month"].map(month_map)

# Adjust year for Oct-Dec, since they belong to the previous hydrological year
df_wrc_long["hydro_year"] = df_wrc_long["year"]
df_wrc_long.loc[df_wrc_long["month"].isin(["Oct", "Nov", "Dec"]), "hydro_year"] -= 1

df_wrc_long["date"] = pd.to_datetime(
    dict(
        year=df_wrc_long["hydro_year"],
        month=df_wrc_long["month_num"],
        day=1
    )
)

df_wrc_long = df_wrc_long.sort_values("date").set_index("date")[["value"]]

# Convert from MCM/month to m3/s
df_wrc_long_converted = df_wrc_long.copy()
seconds_per_month = df_wrc_long_converted.index.days_in_month * 24 * 3600
df_wrc_long_converted['value'] = df_wrc_long_converted['value'] * 1e6 / seconds_per_month

df_wrc_long = df_wrc_long_converted

# Optional: resample to month-end
df_wrc_long = df_wrc_long.resample("ME").mean()

#%% import all WRC datafiles and sum them

folder = r"C:\Users\verwegen\Thesis\Data\Luvuvhu\Comparing"
files = sorted(glob.glob(os.path.join(folder, "A91*.ans")))

cols = [
    "year",
    "Oct", "Nov", "Dec",
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep",
    "annual"
]

dfs_long = []

for f in files:
    df = pd.read_csv(f, delim_whitespace=True, header=None, names=cols)
    
    # Convert years to numbers to melt the dataframe to
    df = df[df["year"].apply(lambda x: str(x).isdigit())]
    df["year"] = df["year"].astype(int)
    
    df_long = df.melt(
        id_vars="year",
        value_vars=[
            "Oct", "Nov", "Dec",
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep"
        ],
        var_name="month",
        value_name="value"
    )
    
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    df_long["month_num"] = df_long["month"].map(month_map)
    
    # Adjust year for Oct-Dec
    df_long["hydro_year"] = df_long["year"]
    df_long.loc[df_long["month"].isin(["Oct", "Nov", "Dec"]), "hydro_year"] -= 1
    
    # Remove data when incorrect
    df_long = df_long[df_long["hydro_year"] > 0]  # no negative values
    df_long["date"] = pd.to_datetime(
        dict(
            year=df_long["hydro_year"],
            month=df_long["month_num"],
            day=1
        )
    )
    
    df_long = df_long.sort_values("date").set_index("date")[["value"]]
    
    # Multiply A91H by 0,5
    if os.path.basename(f).startswith("A91H"):
        df_long["value"] / 2

    dfs_long.append(df_long)

# Sum all dfs
df_sum = pd.concat(dfs_long, axis=1).sum(axis=1).to_frame(name="value")

# Convert from MCM/month to m3/s
seconds_per_month = df_sum.index.days_in_month * 24 * 3600
df_sum["value"] = df_sum["value"] * 1e6 / seconds_per_month

# Resample to month-end
df_sum = df_sum.resample("ME").mean()

df_wrc_long = df_sum.copy()

print(df_sum.head())

#%% import simulated 

path2 = r"C:\Users\verwegen\Thesis\Data\Luvuvhu\Comparing\Simulated_fitted.csv"

df_sim = pd.read_csv(
    path2,
    # parse_dates=[0],   # parse first column as datetime
    # index_col=0        # set first column as index
)

df_sim = df_sim.iloc[:, :2]
df_sim.dropna(inplace=True)
df_sim['baseline_time'] = pd.to_datetime(df_sim['baseline_time'])
df_sim = df_sim.set_index(df_sim['baseline_time'])
del(df_sim['baseline_time'])

df_sim_month = df_sim.resample("ME").mean()

#%%

df_all = (
    df_sim_month.rename(columns={"value": "sim"})
    .join(df_wrc_long.rename(columns={"value": "wrc"}), how="outer")
    .join(df_obs_old_month.rename(columns={"value": "obs"}), how="outer")
)

df_all_clean = df_all.dropna()
df_all_clean = df_all_clean[df_all_clean.index.year != 2000]

#%%

colors1 = ['#0072B2', '#009E73', '#D55E00'] 

fig, ax = plt.subplots(figsize=(12,6))

for col, color in zip(df_all_clean.columns, colors1):
    ax.plot(df_all_clean.index, df_all_clean[col], marker='o', linestyle='-', label=col, color=color)

ax.set_xlabel('Time')
ax.set_ylabel('Discharge')
ax.set_title('Monthly discharge comparison')
ax.legend()
plt.tight_layout()
plt.show()

#%% PLOTTING HYDROLOGICAL YEARS

# Colorblind-safe
colors = ['#009E73','#0072B2','#D55E00']    

# Hydrological year starting in October

hydro_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]

month_names_hydro = ['Oct', 'Nov', 'Dec','Jan', 'Feb', 'Mar',
                     'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep']

months = range(1, 13)

# Resampling
df = df_all_clean.copy()
df['month'] = df.index.month

# Entire period climatology
df_clim_full = df.groupby('month').mean().reindex(hydro_order)
wrc_clim = df.groupby('month')['wrc'].mean().reindex(hydro_order)

periods = {
    "1990-1996": (1990, 1996),
    "1996–1998": (1996, 1998),
    "2005–2007": (2005, 2007),
}

# plot all data compared to eachother before diving into comparing it over subsets
plt.figure(figsize=(12, 6))

for col, color in zip(df_clim_full.columns, colors1):
    plt.plot(months, df_clim_full[col],
             marker='o', linewidth=2, label=col, color=color)

plt.xticks(months, month_names_hydro)
plt.xlabel('Hydrological Year')
plt.ylabel('Mean discharge per month [m³/s]')
plt.title('Monthly Average Discharge (1990–2008)',
          fontsize=14, fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# plot over subset periods

for label, (start, end) in periods.items():

    subset = df[(df.index.year >= start) &
                (df.index.year <= end)]

    clim_period = subset.groupby('month')[['obs', 'baseline']].mean().reindex(hydro_order)

    plt.figure(figsize=(12, 6))

    plt.plot(months, wrc_clim,
             marker='o',linewidth=2,  label='WRC entire period', color = colors[0])

    plt.plot(months, clim_period['obs'], color = colors[1],
             marker='s',linewidth=2,  label=f'Observed {label}')

    plt.plot(months, clim_period['baseline'], color = colors[2],
             marker='^',linewidth=2,  label=f'Simulated baseline {label}')

    plt.xticks(months, month_names_hydro)
    plt.xlabel('Hydrological Year')
    plt.ylabel('Average discharge [m³/s]')
    plt.title(f'Discharge Comparison per Month ({label})', fontsize=14, fontweight='bold')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% big differences over the annual averages visible

# Compute yearly mean discharge
df_yearly = df_all_clean.resample('Y').mean()

plt.figure(figsize=(12,6))

plt.plot(df_yearly.index.year, df_yearly['wrc'], color = colors[0], marker='o', label='WRC')
plt.plot(df_yearly.index.year, df_yearly['obs'], color = colors[1], marker='s', label='Observed')
plt.plot(df_yearly.index.year, df_yearly['baseline'], marker='^',color = colors[2], label='Baseline')

plt.xlabel('Year')
plt.ylabel('Mean yearly discharge [m³/s]')
plt.title('Annual Mean Discharge per Year')
plt.grid(True)
plt.legend()

years = df_yearly.index.year
step = max(1, len(years)//10)  # choose ~10 ticks
plt.xticks(years[::step])  # rotate labels for readability

plt.tight_layout()
plt.show()

#%% FFC

from hydropattern.timeseries import Timeseries

# %%
ts = Timeseries.from_dataframe(df_all, first_dowy=75)
ts.data.head()
# %%
