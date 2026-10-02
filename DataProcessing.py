# %% Required libraries
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Creating the DataFrame and adding all Fred information (One by one since I have to change the columns)
# %% Creat empty Data Frame 
Main = pd.DataFrame({"Date": pd.date_range("2006-01-01", "2026-12-01", freq="MS")})
# %% Federal Funds Effective Rate
DFF = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/DFF.csv", parse_dates=["observation_date"])
DFF = DFF.rename(columns={"observation_date":"Date"})
Main = Main.merge(DFF[["Date", "DFF"]], on="Date", how="left")
# %% Real GDP
RGDP = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/GDPC1.csv", parse_dates=["observation_date"])
RGDP = RGDP.rename(columns={"observation_date":"Date","GDPC1":"Real_GDP"})
Main = Main.merge(RGDP[["Date", "Real_GDP"]], on="Date", how="left")
# %% USA Population
USPOP = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/POPTHM.csv", parse_dates=["observation_date"])
USPOP = USPOP.rename(columns={"observation_date":"Date","POPTHM":"Monthly_US_Pop"})
Main = Main.merge(USPOP[["Date", "Monthly_US_Pop"]], on="Date", how="left")
# %% Median Real earnings
LES = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/LES1252881600Q.csv", parse_dates=["observation_date"])
LES = LES.rename(columns={"observation_date":"Date","LES1252881600Q":"Median_Weekly_Earnings"})
Main = Main.merge(LES[["Date", "Median_Weekly_Earnings"]], on="Date", how="left")
# %% Private Average Hourly Wage
CES = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/CES0500000003.csv", parse_dates=["observation_date"])
CES = CES.rename(columns={"observation_date":"Date","CES0500000003":"Average_Hourly_Wage"})
Main = Main.merge(CES[["Date", "Average_Hourly_Wage"]], on="Date", how="left")
# %% Data CPS transformation
CPI = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/CPI20261002101827_cbf131.csv")
CPI_Long = CPI.melt(id_vars="Year", var_name="Month", value_name=("CPI"))
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CPI_Long["Date"]= pd.to_datetime(CPI_Long["Year"].astype(str) + "-" + CPI_Long["Month"].astype(str),format="%Y-%b")
Main = Main.merge(CPI_Long[["Date", "CPI"]], on="Date", how="left")
# %%Labor Force as levels (Thousands)
LBFL = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/LaborForce_Levels-20260930141900_469142.csv")
LBFL_Long = LBFL.melt(id_vars="Year", var_name="Month",value_name="LBFL")
LBFL_Long["Date"] = pd.to_datetime(LBFL_Long["Year"].astype(str) + "-" + LBFL_Long["Month"].astype(str),format="%Y-%b")
Main = Main.merge(LBFL_Long[["Date", "LBFL"]], on="Date", how="left")
# Now we will do the same for the rest 
# %% Labor Force as Percentage 
LBFP = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/LaborForce_Percentage-20260930141909_3af81d.csv")
LBFP_Long = LBFP.melt(id_vars="Year", var_name="Month",value_name="LBFP")
LBFP_Long["Date"] = pd.to_datetime(LBFP_Long["Year"].astype(str) + "-" + LBFP_Long["Month"].astype(str),format="%Y-%b")
Main = Main.merge(LBFP_Long[["Date", "LBFP"]], on="Date", how="left")
# %% Employment-Population Ratio
EPR = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/EmploymentPopulation_Ratio-20260930141912_2e3bc9.csv")
EPR_Long = EPR.melt(id_vars="Year", var_name="Month",value_name="EPR")
EPR_Long["Date"] = pd.to_datetime(EPR_Long["Year"].astype(str) + "-" + EPR_Long["Month"].astype(str),format="%Y-%b")
Main = Main.merge(EPR_Long[["Date", "EPR"]], on="Date", how="left")
# %% Unemployment Levels
UL = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/Unemployment_Level-20260930141914_298570.csv")
UL_Long = UL.melt(id_vars="Year", var_name="Month",value_name="UL")
UL_Long["Date"] = pd.to_datetime(UL_Long["Year"].astype(str) + "-" + UL_Long["Month"].astype(str),format="%Y-%b")
Main = Main.merge(UL_Long[["Date", "UL"]], on="Date", how="left")
# %% Unemployment Percentage
UP = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/Unemployment_Percentage-20260930141916_744738.csv")
UP_Long = UP.melt(id_vars="Year", var_name="Month",value_name="UP")
UP_Long["Date"] = pd.to_datetime(UP_Long["Year"].astype(str) + "-" + UP_Long["Month"].astype(str),format="%Y-%b")
Main = Main.merge(UP_Long[["Date", "UP"]], on="Date", how="left")
# %% Weekly Unemployed on Average
WEA = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/AverageUnemployedWeeks-20260930141919_39dee8.csv")
WEA_Long = WEA.melt(id_vars="Year", var_name="Month",value_name="WEA")
WEA_Long["Date"] = pd.to_datetime(WEA_Long["Year"].astype(str) + "-" + WEA_Long["Month"].astype(str),format="%Y-%b")
Main = Main.merge(WEA_Long[["Date", "WEA"]], on="Date", how="left")
# %% Not in Labor Force (Thousands)
NLF = pd.read_csv("/home/nonoriri/Documents/GitHub/Forecast_Project/Data/NotinLabor-20260930141921_16ebd7.csv")
NLF_Long = NLF.melt(id_vars="Year", var_name="Month",value_name="NLF")
NLF_Long["Date"] = pd.to_datetime(NLF_Long["Year"].astype(str) + "-" + NLF_Long["Month"].astype(str),format="%Y-%b")
Main = Main.merge(NLF_Long[["Date", "NLF"]], on="Date", how="left")
# %% Data manipulation
Main = Main.set_index("Date")
# %%
plt.plot(Main["EPR"])
plt.xlabel("Date")
plt.ylabel("Employment-Population Ratio")
plt.title("Monthly Employment to Population Ratio in the US 2006-2026")
plt.show()
# %% Summary as an image 
Summary = Main.describe().T.round(2)
plt.figure(figsize=(8, 3))
plt.axis("off")
plt.table(cellText=Summary.values, rowLabels=Summary.index, colLabels=Summary.columns, loc="center")
plt.savefig("Summary.png", dpi=200, bbox_inches="tight")
plt.show()