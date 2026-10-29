# %% Required libraries
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
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
# %% ACF and PACF after removing the dummies
beta = np.linalg.lstsq(det_full, y_full.values, rcond=None)[0]
resid_det = pd.Series(y_full.values - det_full @ beta, index=y_full.index)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
plot_acf(resid_det, lags=24, ax=axes[0])
plot_pacf(resid_det, lags=24, ax=axes[1])
plt.tight_layout()
plt.show()
# %% Compare ARMA orders with the same dummies
# det_full columns: constant, trend, DU_2009, COVID. ARIMA adds its own constant,
# so leave the first column out.
exog = pd.DataFrame(det_full[:, 1:], index=y_full.index,
                    columns=["t", "DU_2009", "COVID"])

orders = [(1, 0, 1), (2, 0, 1), (3, 0, 1), (1, 0, 0)]
fits = {}
rows = []

for order in orders:                                   # one pass per ARMA order
    fit = ARIMA(y_full, exog=exog, order=order, trend="c").fit()
    fits[order] = fit                                  # keep each fitted model for later
    lb = acorr_ljungbox(fit.resid.iloc[12:], lags=[12, 24], return_df=True)["lb_pvalue"]
    rows.append({"order": order, "AIC": fit.aic, "BIC": fit.bic,
                 "LB(12) p": lb.iloc[0], "LB(24) p": lb.iloc[1]})
    print(f"\nARMA{order}")
    print(fits[order].summary()
# %% Residual plots for every candidate
for order in orders:
    resid = fits[order].resid.iloc[12:]          # drop the first 12 (burn-in)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(resid.index, resid, linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"ARMA{order} Residuals (post burn-in)")
    ax.set_xlim(resid.index.min(), resid.index.max())
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot_acf(resid, lags=24, ax=ax)
    ax.set_title(f"ACF of ARMA{order} Residuals")
    fig.tight_layout()
    plt.show()
# %% Quick check(AR(1))
fit = ARIMA(y_full, exog=exog, order=(1, 0, 0), trend="c").fit()

print("AR root (abs):", np.abs(fit.arroots))        # must be > 1
print(acorr_ljungbox(fit.resid.iloc[12:], lags=[12, 24], return_df=True)["lb_pvalue"])   # want > 0.05
# %% Forecast
h = 24
future_idx = pd.date_range(y_full.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")

exog_future = pd.DataFrame({
    "t": np.arange(len(y_full), len(y_full) + h),
    "DU_2009": 1,
    "COVID": 0,
}, index=future_idx)

fc = fit.get_forecast(steps=h, exog=exog_future)
forecast_mean = fc.predicted_mean
conf_int = fc.conf_int()          # columns are named after the series, so use iloc below

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_full.loc["2022-01-01":], label="Observed")
ax.plot(forecast_mean.index, forecast_mean.values, label="Forecast", color="firebrick")
ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                color="firebrick", alpha=0.15, label="95% CI")
ax.set_title(f"Employment-Population Ratio -- {h}-Month Forecast (ARMA(1,0,0) with dummies)")
ax.legend()
plt.tight_layout()
plt.show()

print(pd.concat([forecast_mean.rename("forecast"), conf_int], axis=1).round(3))
# %% AR(1) on the post-COVID sample: fit, diagnostics and forecast
# ---- 1. Sample and fit --------------------------------------------------
start = "2021-07-01"                       # first month after the COVID recovery; change if you want
h = 24                                     # forecast horizon
burn = 3                                   # first residuals to drop (initialization)

y_post = y_full.loc[start:].asfreq("MS")   # monthly frequency so statsmodels does not warn
print(f"Sample: {y_post.index[0].date()} to {y_post.index[-1].date()} ({len(y_post)} obs)\n")

# No dummies here: DU_2009 would be constant (1) and COVID would be 0 in this sample
fit = ARIMA(y_post, order=(1, 0, 0), trend="c").fit()

# ---- 2. Summary and coefficient table ----------------------------------
print(fit.summary())

ci = fit.conf_int()
coef_table = pd.DataFrame({
    "coef": fit.params, "std err": fit.bse, "z": fit.tvalues,
    "P>|z|": fit.pvalues, "[0.025": ci.iloc[:, 0], "0.975]": ci.iloc[:, 1],
}).round(4)
print("\nCoefficient table")
print(coef_table)

long_run_mean = fit.params["const"] / (1 - fit.params["ar.L1"])
print(f"\nImplied long-run mean: {long_run_mean:.3f}")
print(f"AR root (abs, must be > 1): {np.abs(fit.arroots)}")

# ---- 3. Residual tests --------------------------------------------------
resid = fit.resid.iloc[burn:]

lb = acorr_ljungbox(resid, lags=[6, 12], model_df=1, return_df=True)   # model_df=1: one AR parameter
print("\nLjung-Box (H0: no autocorrelation in the residuals)")
print(lb.round(4))

jb_stat, jb_p, skew, kurt = jarque_bera(resid)
print(f"\nJarque-Bera (H0: normal residuals): stat {jb_stat:.3f}, p {jb_p:.3f}, "
      f"skew {skew:.2f}, kurtosis {kurt:.2f}")

# ---- 4. Residual plots --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(resid.index, resid, linewidth=1.2)
axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
axes[0].set_title("AR(1) residuals (post-2021 sample)")
plot_acf(resid, lags=min(24, len(resid) // 2 - 1), ax=axes[1])
axes[1].set_title("ACF of residuals")
fig.tight_layout()
plt.show()

# ---- 5. Forecast --------------------------------------------------------
fc = fit.get_forecast(steps=h)
forecast_mean = fc.predicted_mean
conf_int = fc.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_post, label="Observed")
ax.plot(forecast_mean.index, forecast_mean.values, label="Forecast", color="firebrick")
ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                color="firebrick", alpha=0.15, label="95% CI")
ax.set_title(f"Employment-Population Ratio -- {h}-Month Forecast (AR(1), post-2021 sample)")
ax.legend()
fig.tight_layout()
plt.show()

print("\nForecast table")
print(pd.concat([forecast_mean.rename("forecast"), conf_int], axis=1).round(3))