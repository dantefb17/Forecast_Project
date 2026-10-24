# %% Required libraries
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as stat
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from scipy import stats
# %% Data
Main = pd.read_csv("MainDataSet.csv", index_col="Date", parse_dates=True)
Main["EPR_clean"] = Main["EPR"].ffill(limit=1)
# %%  ADF test 
print("ADF test")
print("H0: the series has a unit root (non-stationary)")
adf_level = adfuller(Main["EPR_clean"], autolag="AIC")
print(f"  statistic = {adf_level[0]:.4f}")
print(f"  p-value   = {adf_level[1]:.4f}")
print(f"  lags used = {adf_level[2]}")
# %% KPSS
print()
print("KPSS test")
print("H0: the series is stationary")
kpss_level = kpss(Main["EPR_clean"], regression="c", nlags="auto")
print(f"  statistic = {kpss_level[0]:.4f}")
print(f"  p-value   = {kpss_level[1]:.4f}")
# The series is stationary
# %% ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
plot_acf(EPR_clean, lags=12, ax=axes[0])
plot_pacf(EPR_clean, lags=12, ax=axes[1])
for ax in axes:
    for seasonal_lag in [4, 8, 12]:
        ax.axvline(seasonal_lag, color="gray", linestyle=":", linewidth=1)
fig.tight_layout()
plt.show()
# %% ARIMA test Initial
auto_model = pm.auto_arima(
    Main["EPR_clean"],
    #seasonal=True,
    #m=4,  # quarterly seasonal period
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
)

print(auto_model.summary())