"""
Financial performance metrics and factor evaluation metrics.

Two groups:
  PERFORMANCE METRICS (pure numpy):
    - sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, win_rate.

  FACTOR EVALUATION METRICS (uses ``scipy.stats.spearmanr``):
    - ic, rolling_ic, information_ratio, quantile_spread.

All functions handle empty / short / edge-case inputs gracefully and return
``0.0`` rather than raising.
"""

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns, rf=0.0):
    """Annualized Sharpe ratio.

    Args:
        returns: Daily returns (list or ndarray).
        rf: Risk-free rate (annualized, default 0.0).

    Returns:
        float: Annualized Sharpe ratio, or 0.0 for <2 returns or zero std.

    Excess return per unit of volatility. GIFT uses this as the primary
    selection criterion when picking the best code sample per iteration.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    std = r.std() * np.sqrt(252)
    if std == 0:
        return 0.0
    mean_annual = r.mean() * 252
    return float((mean_annual - rf) / std)


def sortino_ratio(returns, rf=0.0):
    """Annualized Sortino ratio using downside deviation.

    Args:
        returns: Daily returns (list or ndarray).
        rf: Risk-free rate (annualized, default 0.0).

    Returns:
        float: Annualized Sortino ratio, or 0.0 for <2 returns or zero downside dev.

    Similar to Sharpe but only penalises downside volatility (negative
    returns), which is often more meaningful for portfolio evaluation.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    neg = r[r < 0]
    if len(neg) == 0:
        # No negative returns: downside dev is zero, Sortino is undefined -> 0.0
        return 0.0
    downside_dev = np.sqrt(np.mean(neg ** 2)) * np.sqrt(252)
    if downside_dev == 0:
        return 0.0
    mean_annual = r.mean() * 252
    return float((mean_annual - rf) / downside_dev)


def max_drawdown(returns):
    """Maximum drawdown as a positive percentage.

    Args:
        returns: Daily returns (list or ndarray).

    Returns:
        float: Max drawdown as positive percentage (e.g. 20.0 for 20%).

    Largest peak-to-trough decline of the portfolio, i.e. the worst-case loss.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    return float(abs(drawdown.min())) * 100


def calmar_ratio(returns, rf=0.0):
    """Calmar ratio: annualized return / max drawdown.

    Args:
        returns: Daily returns (list or ndarray).
        rf: Risk-free rate (annualized, default 0.0).

    Returns:
        float: Calmar ratio, or 0.0 if max_drawdown is zero.

    Annualised return per unit of max drawdown — a risk-adjusted measure.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    mdd = max_drawdown(r)
    if mdd == 0:
        return 0.0
    mean_annual = r.mean() * 252 - rf
    return float(mean_annual / (mdd / 100))


def win_rate(returns):
    """Fraction of non-zero returns that are positive.

    Args:
        returns: Daily returns (list or ndarray).

    Returns:
        float: Win rate in [0, 1], or 0.0 if no non-zero returns.

    Fraction of non-zero return days that were positive.
    """
    r = np.asarray(returns, dtype=float)
    non_zero = r[r != 0]
    if len(non_zero) == 0:
        return 0.0
    return float(np.sum(non_zero > 0) / len(non_zero))


# ---------------------------------------------------------------------------
# Factor Evaluation Metrics
# ---------------------------------------------------------------------------

def ic(feature_values, forward_returns, method='spearman'):
    """Information Coefficient: rank correlation between feature and forward return.

    Args:
        feature_values: Feature values (ndarray).
        forward_returns: Forward returns (ndarray).
        method: Correlation method (default 'spearman').

    Returns:
        float: Spearman rank correlation, or 0.0 for <5 pairs or on error.

    The IC measures the predictive power of an LLM-generated feature against
    forward returns and is a core signal in the COT feedback loop.
    """
    fv = np.asarray(feature_values, dtype=float).ravel()
    fr = np.asarray(forward_returns, dtype=float).ravel()
    if len(fv) < 5 or len(fr) < 5:
        return 0.0
    n = min(len(fv), len(fr))
    fv, fr = fv[:n], fr[:n]
    # Guard against NaN/Inf
    mask = np.isfinite(fv) & np.isfinite(fr)
    fv, fr = fv[mask], fr[mask]
    if len(fv) < 5:
        return 0.0
    try:
        corr, _ = spearmanr(fv, fr)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def rolling_ic(feature_values, forward_returns, window=20):
    """Rolling window IC.

    Args:
        feature_values: Feature values (ndarray).
        forward_returns: Forward returns (ndarray).
        window: Rolling window size (default 20).

    Returns:
        ndarray: Array of rolling IC values, length ``max(0, n - window + 1)``.

    Computes IC over sliding windows to track changes in predictive power
    over time.
    """
    fv = np.asarray(feature_values, dtype=float).ravel()
    fr = np.asarray(forward_returns, dtype=float).ravel()
    n = min(len(fv), len(fr))
    if n < window or window < 2:
        return np.array([], dtype=float)
    fv, fr = fv[:n], fr[:n]
    out_len = n - window + 1
    result = np.zeros(out_len, dtype=float)
    for i in range(out_len):
        seg_fv = fv[i:i + window]
        seg_fr = fr[i:i + window]
        # Filter NaN/Inf
        mask = np.isfinite(seg_fv) & np.isfinite(seg_fr)
        seg_fv_c, seg_fr_c = seg_fv[mask], seg_fr[mask]
        if len(seg_fv_c) < 5:
            result[i] = 0.0
            continue
        try:
            corr, _ = spearmanr(seg_fv_c, seg_fr_c)
            result[i] = 0.0 if np.isnan(corr) else float(corr)
        except Exception:
            result[i] = 0.0
    return result


def information_ratio(rolling_ic_series):
    """Information Ratio: mean(IC) / std(IC).

    Args:
        rolling_ic_series: Array of IC values (e.g., from rolling_ic).

    Returns:
        float: Information ratio, or 0.0 for len<2 or zero std.

    Mean(IC) / std(IC); higher = more stable predictive power.
    """
    series = np.asarray(rolling_ic_series, dtype=float).ravel()
    # Filter NaN
    series = series[np.isfinite(series)]
    if len(series) < 2:
        return 0.0
    std = series.std()
    if std == 0:
        return 0.0
    return float(series.mean() / std)


def quantile_spread(feature_values, forward_returns, n_quantiles=5):
    """Quantile spread: mean(top quantile returns) - mean(bottom quantile returns).

    Args:
        feature_values: Feature values (ndarray).
        forward_returns: Forward returns (ndarray).
        n_quantiles: Number of quantile groups (default 5).

    Returns:
        float: Top-minus-bottom spread, or 0.0 for insufficient data.

    Difference between the average return of the top quantile and that of the
    bottom quantile — a measure of the feature's cross-sectional sorting power.
    """
    fv = np.asarray(feature_values, dtype=float).ravel()
    fr = np.asarray(forward_returns, dtype=float).ravel()
    n = min(len(fv), len(fr))
    if n < n_quantiles:
        return 0.0
    fv, fr = fv[:n], fr[:n]
    # Filter NaN/Inf
    mask = np.isfinite(fv) & np.isfinite(fr)
    fv, fr = fv[mask], fr[mask]
    if len(fv) < n_quantiles:
        return 0.0
    # Sort by feature values
    sort_idx = np.argsort(fv)
    sorted_fr = fr[sort_idx]
    # Split into n_quantiles groups
    quantile_size = len(sorted_fr) // n_quantiles
    if quantile_size < 1:
        return 0.0
    bottom = sorted_fr[:quantile_size]
    top = sorted_fr[-quantile_size:]
    return float(np.mean(top) - np.mean(bottom))
