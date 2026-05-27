"""
Market-level regime detector for portfolio optimization.

Computes a 3-dimensional regime vector from the equal-weight portfolio of all
5 stocks:
  [0] trend_direction:  [-1, +1]  (positive = up, negative = down)
  [1] volatility_level: [0, 1]    (higher = more unstable market)
  [2] risk_level:       [0, 1]    (higher = greater risk)

Input: dict of 5 raw states {ticker: 120d_array}. The output becomes part of
the PPO state vector so the agent can perceive the market environment.
"""

import numpy as np

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']


def _extract_closes(s: np.ndarray) -> np.ndarray:
    n = len(s) // 6
    return np.array([s[i * 6] for i in range(n)], dtype=float)


def detect_market_regime(raw_states: dict) -> np.ndarray:
    """Compute 3-dim market regime from the equal-weight portfolio.

    Returns: ``[trend_direction, volatility_level, risk_level]``.
    """
    all_closes = []
    for ticker in TICKERS:
        if ticker in raw_states:
            all_closes.append(_extract_closes(raw_states[ticker]))

    if not all_closes:
        return np.array([0.0, 0.5, 0.0])

    min_len = min(len(c) for c in all_closes)
    if min_len < 5:
        return np.array([0.0, 0.5, 0.0])

    aligned = np.array([c[:min_len] for c in all_closes])
    port_closes = np.mean(aligned, axis=0)

    trend = _trend_direction(port_closes)
    volatility = _volatility_level(port_closes)
    risk = _risk_level(port_closes)

    return np.array([trend, volatility, risk], dtype=float)


def _trend_direction(closes: np.ndarray) -> float:
    """Trend direction: deviation of 5-day MA from the full-period mean, clipped to ``[-1, +1]``."""
    if len(closes) < 5 or np.std(closes) < 1e-8:
        return 0.0
    ma5 = np.mean(closes[-5:])
    ma_all = np.mean(closes)
    trend = (ma5 - ma_all) / (np.mean(closes) * 0.05 + 1e-8)
    return float(np.clip(trend, -1, 1))


def _volatility_level(closes: np.ndarray) -> float:
    """Volatility level: z-score of recent vol vs historical vol, mapped to ``[0, 1]``."""
    if len(closes) < 5:
        return 0.5
    returns = np.diff(closes) / (closes[:-1] + 1e-8)
    recent_vol = np.std(returns[-5:])
    hist_std = np.std(returns) * 0.5 + 1e-10
    z = (recent_vol - np.std(returns)) / hist_std
    return float(np.clip((z + 1) / 3, 0, 1))


def _risk_level(closes: np.ndarray) -> float:
    """Risk level: recent max-drawdown depth, normalized to ``[0, 1]`` (15% drawdown -> 1.0)."""
    if len(closes) < 3:
        return 0.0
    window = closes[-min(10, len(closes)):]
    recent_high = np.max(window)
    current = closes[-1]
    dd = (recent_high - current) / (recent_high + 1e-8)
    return float(np.clip(dd / 0.15, 0, 1))
