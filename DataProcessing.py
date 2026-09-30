# %% Example
import pandas as pd

LBFL = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/LaborForce_Levels-20260930141900_469142.csv")

# Wide -> long
LBFL_Long = LBFL.melt(
    id_vars="Year",
    var_name="Month",
    value_name="Value"
)

# Sort by year, then calendar order of months
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LBFL_Long["Month"] = pd.Categorical(LBFL_Long["Month"], categories=month_order, ordered=True)
LBFL_Long = LBFL_Long.sort_values(["Year", "Month"]).reset_index(drop=True)

print(LBFL_Long.head())