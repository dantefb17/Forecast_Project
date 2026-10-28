# %% Required libraries
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from scipy import stats
from Tests import run_chow_test, _qlr_on_segment, detect_multiple_breaks, adf_dummies, kpss_dummies
# %% Data
Main = pd.read_csv("MainDataSet.csv", index_col="Date", parse_dates=True)
Main["EPR_clean"] = Main["EPR"].ffill(limit=1)
EPR_clean = Main["EPR_clean"]
# %% Plot data
plt.plot(Main["EPR_clean"])
plt.xlabel("Date")
plt.ylabel("Employment-Population Ratio")
plt.title("Monthly Employment to Population Ratio in the US 2006-2026")
plt.show()
# %% Bai-Perron -- Multiple Break Detection
detected_breaks = detect_multiple_breaks(EPR_clean.values, min_size=36) # Set to 3 years 

print(f"\n{'-'*50}")
print(f"Bai-Perron (recursive QLR segmentation)")
print(f"{'-'*50}")
print(f"{'Breaks detected:':<20}{len(detected_breaks):>10}")
for bp in detected_breaks:
    print(f"  index {bp:>4} -> {EPR_clean.index[bp].date()}")
print(f"{'-'*50}")

segment_bounds = [0] + detected_breaks + [len(EPR_clean)]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(EPR_clean.index, EPR_clean.values, color="tab:blue", linewidth=1.2,
         label="Employment to Population Ratio", zorder=3)

for bp in detected_breaks:
    ax.axvline(EPR_clean.index[bp], color="firebrick", linestyle="--",
               linewidth=1, zorder=2)

for i in range(len(segment_bounds) - 1):
    start, end = segment_bounds[i], segment_bounds[i + 1]
    seg_dates = EPR_clean.index[start:end]
    seg_values = EPR_clean.values[start:end]
    seg_mean = seg_values.mean()
    seg_sd = seg_values.std()

    ax.hlines(seg_mean, seg_dates[0], seg_dates[-1],
              color="black", linewidth=1.5, zorder=4,
              label="Segment mean" if i == 0 else None)
    ax.fill_between(seg_dates, seg_mean - 2 * seg_sd, seg_mean + 2 * seg_sd,
                     color="gray", alpha=0.15, zorder=1,
                     label="Segment mean +/- 2 SD" if i == 0 else None)

ax.set_title("Employment-to-Population Ratio with Bai-Perron Breakpoints")
ax.set_xlabel("Date")
ax.set_ylabel("Employment-to-Population Ratio")
ax.set_xlim(EPR_clean.index.min(), EPR_clean.index.max())
ax.legend()
plt.tight_layout()
plt.show()
# %% ADF with dummies
# Full sample, trimmed to the last month with data
y_full = EPR_clean.loc[:"2026-06-01"]
print(y_full.isna().sum())            # must be 0

t = np.arange(len(y_full))

def pos(date):
    """Row number of a date in the series."""
    return y_full.index.get_loc(pd.Timestamp(date))

# 1. Level dummy at 2009: 0 before, 1 from January 2009 on
DU_2009 = (t >= pos("2009-01-01")).astype(int)

# 2. Trend-change dummy at 2014 (only if you decide to keep it):
#    0 before October 2014, then 1, 2, 3, ... after
#DT_2014 = np.where(t >= pos("2014-10-01"), t - pos("2014-10-01"), 0)

# 3. COVID pulse: 1 only during the affected months, 0 everywhere else
COVID = ((y_full.index >= "2020-03-01") & (y_full.index <= "2021-06-01")).astype(int)

# Stack them: constant, linear trend, then the dummies
det_full = np.column_stack([np.ones(len(y_full)), t, DU_2009, COVID])

print(det_full.shape[0] == len(y_full))   # must be True
# %% Test ADF after the setting
adf_stat, lag = adf_dummies(y_full.values, det_full)
print(adf_stat, lag)
# %% Compare values 
rng = np.random.default_rng(123)       # fixed seed so results repeat
reps = 1000                             # start small to check it runs; raise to 1000 later
adf_sim = []

for _ in range(reps):                  # repeat the experiment `reps` times
    e = rng.standard_normal(len(y_full))    # random noise, one value per month
    fake = np.cumsum(e)                # running total of noise = a random walk (has a unit root)
    stat, _ = adf_dummies(fake, det_full)   # run YOUR test on the fake series
    adf_sim.append(stat)               # store its statistic

adf_sim = np.array(adf_sim)

print("5% critical value:", np.percentile(adf_sim, 5))
print("Your statistic   :", adf_stat)
print("Simulated p-value:", (adf_sim <= adf_stat).mean())
# %% KPSS test 
kpss_stat = kpss_dummies(y_full.values, det_full)
print(kpss_stat)
# %% KPSS values 
rng = np.random.default_rng(123)
reps = 1000
kpss_sim = []

for _ in range(reps):
    e = rng.standard_normal(len(y_full))      # stationary fake series
    kpss_sim.append(kpss_dummies(e, det_full))

kpss_sim = np.array(kpss_sim)

print("5% critical value:", np.percentile(kpss_sim, 95))   # note: the 95th percentile
print("Your statistic   :", kpss_stat)
print("Simulated p-value:", (kpss_sim >= kpss_stat).mean())
# %% KPSS with dummies
kpss_stat = kpss_dummies(y_full.values, det_full)
# ... your simulation, printing 0.26, 0.1, 0.01

# %% KPSS check: add the 2014 trend-change dummy
DT_2014 = np.where(t >= pos("2014-10-01"), t - pos("2014-10-01"), 0)
det_alt = np.column_stack([det_full, DT_2014])

kpss_alt = kpss_dummies(y_full.values, det_alt)

rng = np.random.default_rng(123)
kpss_sim_alt = np.array([kpss_dummies(rng.standard_normal(len(y_full)), det_alt)
                         for _ in range(1000)])

print("Statistic        :", kpss_alt)
print("5% critical value:", np.percentile(kpss_sim_alt, 95))
print("Simulated p      :", (kpss_sim_alt >= kpss_alt).mean())
# %% KPSS bandwidth check
for L in [5, 10, 15, 20]:
    stat = kpss_dummies(y_full.values, det_full, nlags=L)
    rng = np.random.default_rng(123)
    sim = np.array([kpss_dummies(rng.standard_normal(len(y_full)), det_full, nlags=L)
                    for _ in range(500)])
    print(f"nlags={L}: statistic {stat:.3f}, 5% cv {np.percentile(sim, 95):.3f}")
# %%  ADF test 
print("ADF test")
print("H0: the series has a unit root (non-stationary)")
adf_level = adfuller(Main["EPR_clean"], autolag="AIC")
print(f"  statistic = {adf_level[0]:.4f}")
print(f"  p-value   = {adf_level[1]:.4f}")
print(f"  lags used = {adf_level[2]}")
# %% KPSS test
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
# %% Other specifications
residuals = pd.Series(auto_model.resid(), index = EPR_clean.index).iloc[12:]
print("Ljung-Box test on residuals")
print("H0: residuals are not autocorrelated (i.e., they look like white noise)")
ljung_box = acorr_ljungbox(residuals, lags=[4, 8, 12], return_df=True)
print(ljung_box)

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(residuals.index, residuals, linewidth=1.2)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_title("ARIMA Residuals (post burn-in)")
ax.set_xlim(residuals.index.min(), residuals.index.max())
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6, 3.5))
plot_acf(residuals, lags=12, ax=ax)
for seasonal_lag in [4, 8, 12]:
    ax.axvline(seasonal_lag, color="gray", linestyle=":", linewidth=1)
ax.set_title("ACF of Residuals")
fig.tight_layout()
plt.show()
# %% Plot forecast
forecast_horizon = 12
forecast_result = auto_model.arima_res_.get_forecast(steps=forecast_horizon)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

future_index = pd.date_range(EPR_clean.index[-1] + pd.offsets.MonthBegin(1),
                             periods=forecast_horizon, freq="MS")

forecast_mean = pd.Series(np.asarray(forecast_mean), index=future_index)
conf_int = pd.DataFrame(np.asarray(conf_int), index=future_index, columns=["lower", "upper"])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(EPR_clean.index, EPR_clean.values, label="Observed")
ax.plot(forecast_mean.index, forecast_mean.values,
        label="Forecast", color="firebrick")
ax.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color="firebrick", alpha=0.15, label="95% CI")
ax.set_title(f"Monthly Employment to Population Ratio -- {forecast_horizon}-Month Forecast")
ax.legend()
plt.tight_layout()
plt.show()

forecast_table = pd.concat([forecast_mean.rename("forecast"), conf_int], axis=1)

print(f"\n{'='*72}")
print(f"{forecast_horizon}-Month Forecast -- ARIMA{auto_model.order}")
print(f"{'='*72}")
print(forecast_table)
print(f"{'='*72}")
# %% Structural test - Chow
break_date = "2008-09-01" # Bankrupt Lehman Brothers
break_index = EPR_clean.index.get_loc(break_date)

chow_result = run_chow_test(EPR_clean.values, break_index, label=f"break at {break_date}")
# %% 3. QLR Test -- An Unknown Break Date. Mya not be needed since data clearly states more than one


#qlr_result = _qlr_on_segment(EPR_clean.values, trim=0.15)

#qlr_break_date = EPR_clean.index[qlr_result[0]].date()
#print(f"QLR's best break point corresponds to: {qlr_break_date}")
#print(f"Max F-statistic: {qlr_result[1]:.4f}")
#print(f"p-value: {qlr_result[2]:.4f}")