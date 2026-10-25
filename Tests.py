#Important tests
import numpy as np
from scipy import stats
# Chow test
def run_chow_test(series, break_index, label=""):
    series = np.asarray(series)
    n = len(series)
    k = 1  # one parameter (the mean) per subsample regression
    y1 = series[:break_index]
    y2 = series[break_index:]

    # Pooled model: one constant fit to the whole series.
    rss_pooled = np.sum((series - series.mean()) ** 2)

    # Unpooled model: a separate constant fit to each subsample.
    rss1 = np.sum((y1 - y1.mean()) ** 2)
    rss2 = np.sum((y2 - y2.mean()) ** 2)
    rss_unpooled = rss1 + rss2

    # Chow F-statistic: how much RSS falls when we let the mean differ
    # across subsamples, relative to how much unexplained variation
    # remains once we do.
    f_stat = ((rss_pooled - rss_unpooled) / k) / (rss_unpooled / (n - 2 * k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
    reject_h0 = p_value < 0.05

    print(f"\n{'-'*50}")
    print(f"Chow Test{': ' + label if label else ''}")
    print(f"H0: no break -- the mean is the same in both subsamples")
    print(f"{'-'*50}")
    print(f"{'F-statistic:':<20}{f_stat:>10.4f}")
    print(f"{'p-value:':<20}{p_value:>10.4f}")
    print(f"{'-'*50}")
    verdict = "Reject H0 -- evidence of a break" if reject_h0 \
        else "Fail to reject H0 -- no evidence of a break"
    print(verdict)

    return {"f_stat": f_stat, "p_value": p_value, "reject_h0": reject_h0}
# QLR test
def _qlr_on_segment(segment, trim=0.15):
    
    n = len(segment)
    lo = int(n * trim)
    hi = int(n * (1 - trim))
    if hi - lo < 2:
        return None

    best_F = -np.inf
    best_t = None
    for t in range(lo, hi):
        y1 = segment[:t]
        y2 = segment[t:]
        rss_pooled = np.sum((segment - segment.mean()) ** 2)
        rss_unpooled = np.sum((y1 - y1.mean()) ** 2) + np.sum((y2 - y2.mean()) ** 2)
        F = ((rss_pooled - rss_unpooled) / 1) / (rss_unpooled / (n - 2))
        if F > best_F:
            best_F, best_t = F, t

    p_value = 1 - stats.f.cdf(best_F, 1, n - 2)
    return best_t, best_F, p_value


def detect_multiple_breaks(series, min_size=12, trim=0.15, alpha=0.0005):

    series = np.asarray(series)
    breaks = []

    def recurse(start, end):
        segment = series[start:end]
        if end - start < 2 * min_size:
            return

        result = _qlr_on_segment(segment, trim=trim)
        if result is None:
            return

        local_t, f_stat, p_value = result
        global_break = start + local_t

        # Reject candidates too close to either edge of this segment
        # -- symmetric with the min_size check in run_qlr_test()'s
        # trim logic, but enforced in RAW observation counts here
        # rather than as a fraction, since segments shrink as the
        # recursion proceeds.
        if local_t < min_size or (len(segment) - local_t) < min_size:
            return

        if p_value < alpha:
            breaks.append(global_break)
            recurse(start, global_break)
            recurse(global_break, end)

    recurse(0, len(series))
    return sorted(breaks)