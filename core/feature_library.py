"""
Feature Library for Portfolio PPO

Pure Python + NumPy implementation of 20+ financial indicators with:
- INDICATOR_REGISTRY: name -> {fn, output_dim, default_params, param_ranges, theme}
- build_revise_state(): closure-based assembler from JSON selections to callable
- NormalizedIndicator: Z-score normalization wrapper

Design decisions:
- Pure Python + NumPy only (no ta-lib / pandas_ta dependency).
- Closure-based assembly, no exec/eval.
- Z-score normalization wrapper available for indicator outputs.
- Parameterized indicators with validated ranges (clipped to ``param_ranges``).
- NaN/Inf guards on every indicator and on the assembled closure.

State layout (120d interleaved, from regime_detector.py):
  s[i*6 + 0] = close,  s[i*6 + 1] = open,
  s[i*6 + 2] = high,   s[i*6 + 3] = low,
  s[i*6 + 4] = volume, s[i*6 + 5] = adj_close
  for i = 0..19 (20 trading days)

The four indicator themes are TREND (RSI/MACD/EMA cross/momentum/ROC),
VOLATILITY (Bollinger/ATR/volatility/skew/kurtosis), MEAN_REVERSION
(stochastic/Williams %R/CCI), and VOLUME (OBV/volume ratio/ADX). On top of
these registry indicators, the module also exposes nine "building block"
functions that LLM-generated code can import directly.

Note: ``validate_selection``, ``screen_features``, ``assess_stability``, and
``_dedup_by_base_indicator`` have been moved to ``gift_controller.py`` with
portfolio-aware logic.
"""

import numpy as np
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helper: state extraction from the 120-d interleaved state vector.
# ---------------------------------------------------------------------------

def _extract_ohlcv(s: np.ndarray):
    """Extract OHLCV arrays from 120d interleaved state.

    Returns: ``(closes, opens, highs, lows, volumes)`` as float arrays. The
    input ``s`` follows the layout
    ``[close, open, high, low, volume, adj_close] * 20 days``.
    """
    n = len(s) // 6
    closes = np.array([s[i * 6 + 0] for i in range(n)], dtype=float)
    opens = np.array([s[i * 6 + 1] for i in range(n)], dtype=float)
    highs = np.array([s[i * 6 + 2] for i in range(n)], dtype=float)
    lows = np.array([s[i * 6 + 3] for i in range(n)], dtype=float)
    volumes = np.array([s[i * 6 + 4] for i in range(n)], dtype=float)
    return closes, opens, highs, lows, volumes


# ---------------------------------------------------------------------------
# Helper: moving averages (EMA and SMA).
# ---------------------------------------------------------------------------

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average using numpy convolution.

    Uses exponential decay weights for a proper EMA approximation.
    """
    if len(data) < period or period < 1:
        return np.full_like(data, data[-1] if len(data) > 0 else 0.0)
    alpha = 2.0 / (period + 1.0)
    weights = np.array([(1 - alpha) ** i for i in range(period)])[::-1]
    weights = weights / weights.sum()
    convolved = np.convolve(data, weights, mode='full')[:len(data)]
    return convolved


def _sma(data: np.ndarray, period: int) -> float:
    """Simple moving average of the last ``period`` values."""
    if len(data) < period or period < 1:
        return float(np.mean(data)) if len(data) > 0 else 0.0
    return float(np.mean(data[-period:]))


# ---------------------------------------------------------------------------
# TREND theme indicators (5): RSI, MACD, EMA cross, Momentum, ROC.
# Capture the direction and strength of price trends.
# ---------------------------------------------------------------------------

def compute_rsi(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Wilder's RSI normalized to [0, 1].

    Returns shape ``(1,)``. Neutral default = 0.5 on insufficient data.
    RSI > 0.7 indicates overbought, RSI < 0.3 indicates oversold.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.5])
    deltas = np.diff(closes[-(window + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) + 1e-10
    rs = avg_gain / avg_loss
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return np.array([rsi_val / 100.0])


def compute_macd(s: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD line, Signal line, Histogram.

    Returns shape ``(3,)``: MACD line, signal line, histogram. Output is
    normalised by price; returns zeros on insufficient data.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < slow:
        return np.zeros(3)
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    macd_val = float(macd_line[-1])

    # Signal line: EMA of MACD line (use last `signal` values)
    if len(macd_line) >= signal:
        signal_val = float(_ema(macd_line, signal)[-1])
    else:
        signal_val = 0.0

    histogram = macd_val - signal_val
    # Normalize by recent price to get reasonable scale
    price = closes[-1] if closes[-1] != 0 else 1.0
    return np.array([macd_val / price, signal_val / price, histogram / price])


def compute_ema_cross(s: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    """EMA crossover signal: (EMA_fast - EMA_slow) / price.

    Returns shape ``(1,)``. Positive = bullish cross, negative = bearish.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < slow:
        return np.array([0.0])
    ema_fast_val = float(_ema(closes, fast)[-1])
    ema_slow_val = float(_ema(closes, slow)[-1])
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    cross = (ema_fast_val - ema_slow_val) / price
    return np.array([cross])


def compute_momentum(s: np.ndarray, window: int = 10) -> np.ndarray:
    """Rate of change / momentum: (close[-1] - close[-window]) / close[-window].

    Returns shape ``(1,)``.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    denom = closes[-window - 1]
    if abs(denom) < 1e-10:
        return np.array([0.0])
    mom = (closes[-1] - closes[-window - 1]) / denom
    return np.array([float(mom)])


# ---------------------------------------------------------------------------
# VOLATILITY theme indicators (3): Bollinger Bands, ATR, realised volatility.
# Quantify the magnitude of price moves and overall risk.
# ---------------------------------------------------------------------------

def compute_bollinger(s: np.ndarray, window: int = 20, num_std: float = 2.0) -> np.ndarray:
    """Bollinger Band: upper, middle, lower (normalized by price).

    Returns shape ``(3,)``. Reflects price position relative to its statistical
    volatility band.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.zeros(3)
    recent = closes[-window:]
    sma = np.mean(recent)
    std = np.std(recent, ddof=1) + 1e-10
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    upper = (sma + num_std * std - closes[-1]) / price
    middle = (sma - closes[-1]) / price
    lower = (sma - num_std * std - closes[-1]) / price
    return np.array([upper, middle, lower])


def compute_atr(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Average True Range normalized by price.

    Returns shape ``(1,)``. Often used to size stops or assess risk.
    """
    closes, _, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    atr_val = np.mean(tr[-window:])
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([float(atr_val / price)])


def compute_volatility(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling standard deviation of returns.

    Returns shape ``(1,)``.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    returns = np.diff(closes[-(window + 1):])
    vol = np.std(returns) if len(returns) > 0 else 0.0
    return np.array([float(vol)])


# ---------------------------------------------------------------------------
# MEAN_REVERSION theme indicators (3): Stochastic, Williams %R, CCI.
# Detect mean-reversion signals once price has deviated from its mean.
# ---------------------------------------------------------------------------

def compute_stochastic(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Stochastic Oscillator %K and %D.

    %K = (close - low_N) / (high_N - low_N) * 100
    %D = SMA(%K, 3) (approximated)

    Returns shape ``(2,)``. Values normalised to ``[0, 1]``.
    """
    closes, _, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.5, 0.5])
    recent_high = np.max(highs[-window:])
    recent_low = np.min(lows[-window:])
    price_range = recent_high - recent_low
    if abs(price_range) < 1e-10:
        return np.array([0.5, 0.5])
    pct_k = (closes[-1] - recent_low) / price_range
    # Approximate %D as average of recent %K values
    if len(closes) >= window + 2:
        pk_values = []
        for offset in range(min(3, len(closes) - window)):
            h = np.max(highs[-(window + offset):len(highs) - offset if offset > 0 else len(highs)])
            l = np.min(lows[-(window + offset):len(lows) - offset if offset > 0 else len(lows)])
            r = h - l
            if abs(r) > 1e-10:
                pk_values.append((closes[-(1 + offset)] - l) / r)
            else:
                pk_values.append(0.5)
        pct_d = np.mean(pk_values)
    else:
        pct_d = pct_k
    return np.array([float(pct_k), float(pct_d)])


def compute_williams_r(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Williams %R: (high_N - close) / (high_N - low_N) * -100, normalized to [0, 1].

    Returns shape ``(1,)``. 0 = overbought (near recent high), 1 = oversold
    (near recent low).
    """
    closes, _, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.5])
    recent_high = np.max(highs[-window:])
    recent_low = np.min(lows[-window:])
    price_range = recent_high - recent_low
    if abs(price_range) < 1e-10:
        return np.array([0.5])
    wr = (recent_high - closes[-1]) / price_range
    return np.array([float(wr)])  # [0, 1] where 0=overbought, 1=oversold


def compute_cci(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Commodity Channel Index normalized to [-1, 1] range.

    Returns shape ``(1,)``.
    """
    closes, opens, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.0])
    typical_prices = (highs + lows + closes) / 3.0
    recent_tp = typical_prices[-window:]
    sma_tp = np.mean(recent_tp)
    mean_dev = np.mean(np.abs(recent_tp - sma_tp)) + 1e-10
    cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_dev)
    # Normalize to roughly [-1, 1] range (CCI can be very large)
    return np.array([float(np.clip(cci / 200.0, -1.0, 1.0))])


# ---------------------------------------------------------------------------
# VOLUME theme indicators (3): OBV, Volume Ratio, ADX.
# Validate trend strength from a volume perspective.
# ---------------------------------------------------------------------------

def compute_obv(s: np.ndarray) -> np.ndarray:
    """On-Balance Volume normalized by total volume.

    Returns shape ``(1,)``. Measures buy/sell pressure via cumulative volume on
    up vs down days.
    """
    closes, _, _, _, volumes = _extract_ohlcv(s)
    if len(closes) < 2:
        return np.array([0.0])
    total_vol = np.sum(volumes) + 1e-10
    obv = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
    return np.array([float(obv / total_vol)])


def compute_volume_ratio(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Current volume / average volume over window.

    Returns shape ``(1,)``. Values above 1 indicate above-average volume.
    """
    closes, _, _, _, volumes = _extract_ohlcv(s)
    if len(volumes) < window:
        return np.array([1.0])
    avg_vol = np.mean(volumes[-window:]) + 1e-10
    return np.array([float(volumes[-1] / avg_vol)])


def compute_adx(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Average Directional Index (simplified).

    Returns shape ``(1,)``. Value in ``[0, 1]`` where 1 = strong trend. Does
    not distinguish direction, only strength.
    """
    closes, _, highs, lows, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])

    # True range
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )

    # Directional movement
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smooth with window
    if len(tr) < window:
        return np.array([0.0])
    atr_val = np.mean(tr[-window:]) + 1e-10
    plus_di = 100.0 * np.mean(plus_dm[-window:]) / atr_val
    minus_di = 100.0 * np.mean(minus_dm[-window:]) / atr_val

    denom = plus_di + minus_di + 1e-10
    dx = 100.0 * abs(plus_di - minus_di) / denom
    # Normalize to [0, 1]
    return np.array([float(np.clip(dx / 100.0, 0.0, 1.0))])


# ---------------------------------------------------------------------------
# EXTENDED indicators (7): ROC, SMA cross, DEMA, skewness, kurtosis,
# Williams Alligator, TSF. Add coverage beyond the base set.
# ---------------------------------------------------------------------------

def compute_roc(s: np.ndarray, window: int = 10) -> np.ndarray:
    """Rate of Change: (close[-1] - close[-window]) / close[-window].

    Returns shape ``(1,)``.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    denom = closes[-window - 1]
    if abs(denom) < 1e-10:
        return np.array([0.0])
    roc = (closes[-1] - denom) / denom
    return np.array([float(roc)])


def compute_sma_cross(s: np.ndarray, fast: int = 10, slow: int = 30) -> np.ndarray:
    """SMA crossover signal: (SMA_fast - SMA_slow) / price.

    Returns shape ``(1,)``. Positive = short MA crosses above long MA
    (bullish); negative = bearish.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < slow:
        return np.array([0.0])
    sma_fast = np.mean(closes[-fast:])
    sma_slow = np.mean(closes[-slow:])
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([float((sma_fast - sma_slow) / price)])


def compute_dema(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Double EMA: 2*EMA - EMA(EMA), normalized by price.

    Returns shape ``(1,)``. Reacts faster than a plain EMA (less lag).
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.0])
    ema1 = _ema(closes, window)
    ema2 = _ema(ema1, window)
    dema_val = 2.0 * ema1[-1] - ema2[-1]
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([float(dema_val / price)])


def compute_skewness(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Return distribution skewness over window.

    Returns shape ``(1,)``. Positive = right-skew (extreme positive returns);
    negative = left-skew (extreme negative returns).
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    returns = np.diff(closes[-(window + 1):])
    if len(returns) < 3:
        return np.array([0.0])
    std = np.std(returns, ddof=1) + 1e-10
    mean = np.mean(returns)
    skew = np.mean(((returns - mean) / std) ** 3)
    return np.array([float(np.clip(skew, -5.0, 5.0))])


def compute_kurtosis(s: np.ndarray, window: int = 20) -> np.ndarray:
    """Return distribution kurtosis over window (excess kurtosis).

    Returns shape ``(1,)``. Positive = fat tails (more extreme events);
    negative = thin tails.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window + 1:
        return np.array([0.0])
    returns = np.diff(closes[-(window + 1):])
    if len(returns) < 4:
        return np.array([0.0])
    std = np.std(returns, ddof=1) + 1e-10
    mean = np.mean(returns)
    kurt = np.mean(((returns - mean) / std) ** 4) - 3.0  # excess kurtosis
    return np.array([float(np.clip(kurt, -5.0, 10.0))])


def compute_williams_alligator(s: np.ndarray) -> np.ndarray:
    """Williams Alligator: jaw (13), teeth (8), lips (5) SMAs, normalized by price.

    Returns shape ``(3,)``. The three lines separating indicate trend
    formation; intertwined lines indicate consolidation.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < 13:
        return np.zeros(3)
    jaw = float(np.mean(closes[-13:]))
    teeth = float(np.mean(closes[-8:]))
    lips = float(np.mean(closes[-5:]))
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([jaw / price, teeth / price, lips / price])


def compute_tsf(s: np.ndarray, window: int = 14) -> np.ndarray:
    """Time Series Forecast: linear regression slope * window + intercept, normalized.

    Returns shape ``(1,)``. Reflects the direction and strength of the linear
    price trend.
    """
    closes, _, _, _, _ = _extract_ohlcv(s)
    if len(closes) < window:
        return np.array([0.0])
    recent = closes[-window:]
    x = np.arange(window, dtype=float)
    slope = np.polyfit(x, recent, 1)[0]
    price = closes[-1] if abs(closes[-1]) > 1e-8 else 1.0
    return np.array([float(slope * window / price)])


# ---------------------------------------------------------------------------
# Building-block functions importable from LLM-generated code.
# ---------------------------------------------------------------------------
# Example LLM ``revise_state`` import:
#     from feature_library import (
#         compute_relative_momentum, compute_realized_volatility, ...
#     )
# These nine functions give the LLM well-typed building blocks so that it does
# not have to reimplement financial primitives from scratch.

def compute_relative_momentum(prices: np.ndarray, window: int = 20) -> float:
    """Return of a single stock over ``window`` days.

    Used to identify relative outperformers.
    """
    if len(prices) < window + 1 or prices[-window - 1] == 0:
        return 0.0
    return float((prices[-1] - prices[-window - 1]) / abs(prices[-window - 1]))


def compute_cross_sectional_rank(values: list) -> float:
    """Rank of a single value among all stocks' values. Returns [0, 1].

    Note: portfolio-level only; do not call inside ``revise_state(s)`` since
    that function operates on a single stock.
    """
    if not values or len(values) < 2:
        return 0.5
    target = values[0]
    rank = sum(1 for v in values if v <= target)
    return float(rank / len(values))


def compute_realized_volatility(returns: np.ndarray, window: int = 20) -> float:
    """Realised volatility (std of returns) over ``window`` days."""
    if len(returns) < window:
        window = len(returns)
    if window < 2:
        return 0.0
    return float(np.std(returns[-window:], ddof=1))


def compute_downside_risk(returns: np.ndarray, window: int = 20) -> float:
    """Downside semi-deviation over ``window`` days.

    Focuses on the loss side; uses only negative returns.
    """
    if len(returns) < window:
        window = len(returns)
    if window < 2:
        return 0.0
    neg = returns[-window:]
    neg = neg[neg < 0]
    if len(neg) < 2:
        return 0.0
    return float(np.sqrt(np.mean(neg ** 2)))


def compute_beta(returns: np.ndarray, market_returns: np.ndarray,
                 window: int = 20) -> float:
    """Rolling beta to market (equal-weight portfolio).

    ``beta > 1`` -> stock more volatile than market; ``< 1`` -> less volatile.
    """
    n = min(len(returns), len(market_returns), window)
    if n < 5:
        return 1.0
    r = returns[-n:]
    m = market_returns[-n:]
    var_m = np.var(m)
    if var_m < 1e-10:
        return 1.0
    cov = np.mean((r - np.mean(r)) * (m - np.mean(m)))
    return float(cov / var_m)


def compute_multi_horizon_momentum(prices: np.ndarray,
                                   windows: list = None) -> np.ndarray:
    """Momentum at multiple time horizons (default windows: 5/10/20 days)."""
    if windows is None:
        windows = [5, 10, 20]
    result = []
    for w in windows:
        if len(prices) > w and prices[-w - 1] != 0:
            result.append((prices[-1] - prices[-w - 1]) / abs(prices[-w - 1]))
        else:
            result.append(0.0)
    return np.array(result, dtype=float)


def compute_zscore_price(prices: np.ndarray, window: int = 20) -> float:
    """Z-score of the current price vs the N-day mean.

    Positive = price above mean; negative = below mean. Useful as a
    mean-reversion signal.
    """
    if len(prices) < window:
        return 0.0
    seg = prices[-window:]
    mean_val = np.mean(seg)
    std_val = np.std(seg, ddof=1) + 1e-10
    return float(np.clip((prices[-1] - mean_val) / std_val, -3, 3))


def compute_mean_reversion_signal(prices: np.ndarray, window: int = 20) -> float:
    """Mean-reversion strength: how far price deviated and started reverting.

    Captured as the difference between the prior z-score and the current
    z-score.
    """
    if len(prices) < window + 2:
        return 0.0
    seg_prev = prices[-window - 1:-1]
    mean_prev = np.mean(seg_prev)
    std_prev = np.std(seg_prev, ddof=1) + 1e-10
    z_prev = (prices[-2] - mean_prev) / std_prev
    seg_curr = prices[-window:]
    mean_curr = np.mean(seg_curr)
    std_curr = np.std(seg_curr, ddof=1) + 1e-10
    z_current = (prices[-1] - mean_curr) / std_curr
    return float(np.clip(z_prev - z_current, -3, 3))


def compute_turnover_ratio(volumes: np.ndarray, window: int = 20) -> float:
    """Current volume divided by the average volume over ``window`` days.

    Useful for detecting volume spikes / contractions (a proxy for liquidity).
    """
    if len(volumes) < window + 1:
        return 1.0
    avg = np.mean(volumes[-window - 1:-1]) + 1e-10
    return float(volumes[-1] / avg)


BUILDING_BLOCKS = [
    ('compute_relative_momentum', 'prices, window=20', 1,
     "Excess return vs window-average. Identifies outperforming stocks."),
    ('compute_cross_sectional_rank', 'values', 1,
     "Rank among all stocks [0,1]. Portfolio-level only, not in revise_state."),
    ('compute_realized_volatility', 'returns, window=20', 1,
     "Realized volatility. Measure individual stock risk."),
    ('compute_downside_risk', 'returns, window=20', 1,
     "Downside semi-deviation. Measure downside risk."),
    ('compute_beta', 'returns, market_returns, window=20', 1,
     "Beta to equal-weight portfolio. Systemic risk exposure."),
    ('compute_multi_horizon_momentum', 'prices, windows=[5,10,20]', 3,
     "Multi-period momentum. Capture trends at multiple scales."),
    ('compute_zscore_price', 'prices, window=20', 1,
     "Price z-score vs N-day mean. Mean reversion signal."),
    ('compute_mean_reversion_signal', 'prices, window=20', 1,
     "Mean reversion strength. Identify overextended prices."),
    ('compute_turnover_ratio', 'volumes, window=20', 1,
     "Volume ratio. Liquidity detection."),
]


# ---------------------------------------------------------------------------
# INDICATOR REGISTRY (parameterised with validated ranges).
# Each entry records function, output dim, default params, param ranges, theme.
# ---------------------------------------------------------------------------

INDICATOR_REGISTRY: Dict[str, dict] = {
    # --- TREND theme (5) ---
    'RSI': {
        'fn': compute_rsi,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 60)},
        'theme': 'trend',
    },
    'MACD': {
        'fn': compute_macd,
        'output_dim': 3,
        'default_params': {'fast': 12, 'slow': 26, 'signal': 9},
        'param_ranges': {'fast': (5, 20), 'slow': (15, 60), 'signal': (3, 15)},
        'theme': 'trend',
    },
    'EMA_Cross': {
        'fn': compute_ema_cross,
        'output_dim': 1,
        'default_params': {'fast': 12, 'slow': 26},
        'param_ranges': {'fast': (5, 20), 'slow': (15, 60)},
        'theme': 'trend',
    },
    'Momentum': {
        'fn': compute_momentum,
        'output_dim': 1,
        'default_params': {'window': 10},
        'param_ranges': {'window': (5, 60)},
        'theme': 'trend',
    },
    'ROC': {
        'fn': compute_roc,
        'output_dim': 1,
        'default_params': {'window': 10},
        'param_ranges': {'window': (5, 60)},
        'theme': 'trend',
    },
    # --- VOLATILITY theme (3) ---
    'Bollinger': {
        'fn': compute_bollinger,
        'output_dim': 3,
        'default_params': {'window': 20, 'num_std': 2.0},
        'param_ranges': {'window': (10, 40), 'num_std': (1.0, 3.0)},
        'theme': 'volatility',
    },
    'ATR': {
        'fn': compute_atr,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'volatility',
    },
    'Volatility': {
        'fn': compute_volatility,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 60)},
        'theme': 'volatility',
    },
    # --- MEAN_REVERSION theme (3) ---
    'Stochastic': {
        'fn': compute_stochastic,
        'output_dim': 2,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'mean_reversion',
    },
    'Williams_R': {
        'fn': compute_williams_r,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'mean_reversion',
    },
    'CCI': {
        'fn': compute_cci,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 30)},
        'theme': 'mean_reversion',
    },
    # --- VOLUME theme (3) ---
    'OBV': {
        'fn': compute_obv,
        'output_dim': 1,
        'default_params': {},
        'param_ranges': {},
        'theme': 'volume',
    },
    'Volume_Ratio': {
        'fn': compute_volume_ratio,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 30)},
        'theme': 'volume',
    },
    'ADX': {
        'fn': compute_adx,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'volume',
    },
    # --- EXTENDED indicators (7 more = 21 total) ---
    'SMA_Cross': {
        'fn': compute_sma_cross,
        'output_dim': 1,
        'default_params': {'fast': 10, 'slow': 30},
        'param_ranges': {'fast': (5, 20), 'slow': (15, 60)},
        'theme': 'trend',
    },
    'DEMA': {
        'fn': compute_dema,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 60)},
        'theme': 'trend',
    },
    'Skewness': {
        'fn': compute_skewness,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 60)},
        'theme': 'volatility',
    },
    'Kurtosis': {
        'fn': compute_kurtosis,
        'output_dim': 1,
        'default_params': {'window': 20},
        'param_ranges': {'window': (5, 60)},
        'theme': 'volatility',
    },
    'Williams_Alligator': {
        'fn': compute_williams_alligator,
        'output_dim': 3,
        'default_params': {},
        'param_ranges': {},
        'theme': 'trend',
    },
    'TSF': {
        'fn': compute_tsf,
        'output_dim': 1,
        'default_params': {'window': 14},
        'param_ranges': {'window': (5, 30)},
        'theme': 'trend',
    },
}


# ---------------------------------------------------------------------------
# Z-score normalisation wrapper: standardises indicator outputs.
# ---------------------------------------------------------------------------

class NormalizedIndicator:
    """Wraps an indicator function with Z-score normalization.

    Usage:
        ni = NormalizedIndicator(compute_rsi, {'window': 14}, mean=0.5, std=0.1)
        result = ni(raw_state)  # applies (raw - mean) / (std + 1e-8)
    """

    def __init__(self, fn: Callable, params: dict,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        self.fn = fn
        self.params = params
        self.mean = mean
        self.std = std

    def __call__(self, raw_state: np.ndarray) -> np.ndarray:
        raw = self.fn(raw_state, **self.params)
        if self.mean is not None and self.std is not None:
            return (raw - self.mean) / (self.std + 1e-8)
        return raw


# ---------------------------------------------------------------------------
# Closure-based assembler (no exec/eval).
# ---------------------------------------------------------------------------

def build_revise_state(selection: List[Dict]) -> Callable:
    """Build a closure that computes all selected features.

    Closure-based assembly, no exec/eval. Parameters are clipped to the
    registered ``param_ranges``. NaN/Inf in any indicator output is replaced
    with zeros to keep downstream computation numerically safe.

    Args:
        selection: [{"indicator": "RSI", "params": {"window": 14}}, ...]

    Returns:
        Callable that takes ``raw_state`` (120-d) and returns a 1-D feature
        array.
    """
    funcs = []
    output_dims = []

    for item in selection:
        name = item.get('indicator', '')
        params = dict(item.get('params', {}))  # copy to avoid mutation

        if name not in INDICATOR_REGISTRY:
            continue  # silently skip unknown indicators

        entry = INDICATOR_REGISTRY[name]

        # Merge defaults with user-specified params (user overrides defaults)
        merged = dict(entry['default_params'])
        merged.update(params)

        # Clip params to the registered ranges.
        for pk, pv in merged.items():
            if pk in entry['param_ranges']:
                lo, hi = entry['param_ranges'][pk]
                merged[pk] = type(pv)(np.clip(pv, lo, hi))  # preserve type

        funcs.append((entry['fn'], merged))
        output_dims.append(entry['output_dim'])

    if not funcs:
        # Fallback: return zeros(3) when no valid indicators
        def revise_state_fallback(raw_state: np.ndarray) -> np.ndarray:
            return np.zeros(3)
        return revise_state_fallback

    # Capture funcs and output_dims in closure
    _funcs = funcs
    _output_dims = output_dims

    def revise_state(raw_state: np.ndarray) -> np.ndarray:
        features = []
        for idx, (fn, params) in enumerate(_funcs):
            try:
                result = fn(raw_state, **params)
                # Ensure result is 1D numpy array
                if not isinstance(result, np.ndarray):
                    result = np.atleast_1d(np.array(result, dtype=float))
                if result.ndim != 1:
                    result = result.flatten()
                # NaN/Inf guard.
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    result = np.zeros(_output_dims[idx])
                features.append(result)
            except Exception:
                # Graceful fallback: zeros of correct dimension
                features.append(np.zeros(_output_dims[idx]))

        if not features:
            return np.zeros(3)
        return np.concatenate(features)

    return revise_state
