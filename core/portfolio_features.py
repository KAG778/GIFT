"""
Portfolio-level feature indicators.

Cross-stock indicators that operate on all stocks simultaneously. Unlike
the per-stock indicators in ``feature_library.py``, every function here
consumes the full portfolio state. Each function receives:
  * ``raw_states``: dict mapping ``ticker`` to a 120-d state array (insertion
    order matches the environment's ticker order and weight vector).
  * ``current_weights``: (n_stocks + 1)-d weight vector (stocks + cash).
"""

import numpy as np
from typing import Callable


def _extract_closes(s: np.ndarray) -> np.ndarray:
    n = len(s) // 6
    return np.array([s[i * 6] for i in range(n)], dtype=float)


def _extract_volumes(s: np.ndarray) -> np.ndarray:
    n = len(s) // 6
    return np.array([s[i * 6 + 4] for i in range(n)], dtype=float)


def compute_momentum_rank(raw_states: dict, window: int = 20,
                          current_weights: np.ndarray = None) -> np.ndarray:
    """Rank each stock by its past-N day return. n dims, normalized to ``[0, 1]``.

    The top-ranked stock receives ``1.0``, the bottom-ranked ``1/n``.
    """
    n = len(raw_states)
    ranks = []
    for ticker in raw_states:
        closes = _extract_closes(raw_states[ticker])
        if len(closes) < window + 1 or closes[-window - 1] == 0:
            ranks.append(0.0)
            continue
        ret = (closes[-1] - closes[-window - 1]) / abs(closes[-window - 1])
        ranks.append(ret)
    ranks = np.array(ranks)
    order = np.argsort(ranks)[::-1]
    result = np.zeros(n)
    for rank_idx, ticker_idx in enumerate(order):
        result[ticker_idx] = (n - rank_idx) / float(n)
    return result


def compute_rolling_correlation(raw_states: dict, window: int = 60,
                                current_weights: np.ndarray = None) -> np.ndarray:
    """Pairwise rolling correlation. n*(n-1)/2 dims (n choose 2).

    Measures co-movement across the stocks; higher correlation implies
    weaker diversification benefit.
    """
    keys = list(raw_states.keys())
    n = len(keys)
    all_closes = {}
    for ticker in keys:
        closes = _extract_closes(raw_states[ticker])
        returns = np.diff(closes) / (closes[:-1] + 1e-10)
        all_closes[ticker] = returns[-window:] if len(returns) >= window else returns

    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            r1 = all_closes[keys[i]]
            r2 = all_closes[keys[j]]
            n_seg = min(len(r1), len(r2))
            if n_seg < 5:
                corrs.append(0.0)
                continue
            r1_seg, r2_seg = r1[-n_seg:], r2[-n_seg:]
            std1, std2 = np.std(r1_seg), np.std(r2_seg)
            if std1 < 1e-10 or std2 < 1e-10:
                corrs.append(0.0)
                continue
            corr = float(np.mean((r1_seg - np.mean(r1_seg)) * (r2_seg - np.mean(r2_seg))) / (std1 * std2))
            corrs.append(np.clip(corr, -1, 1))
    return np.array(corrs)


def compute_relative_strength(raw_states: dict, window: int = 20,
                              current_weights: np.ndarray = None) -> np.ndarray:
    """Excess return of each stock relative to the equal-weight basket. n dims.

    Positive = stock outperforms the basket; negative = underperforms.
    """
    all_rets = []
    for ticker in raw_states:
        closes = _extract_closes(raw_states[ticker])
        if len(closes) < window + 1 or closes[-window - 1] == 0:
            all_rets.append(0.0)
            continue
        ret = (closes[-1] - closes[-window - 1]) / abs(closes[-window - 1])
        all_rets.append(ret)
    all_rets = np.array(all_rets)
    basket_ret = np.mean(all_rets)
    return all_rets - basket_ret


def compute_portfolio_volatility(raw_states: dict, window: int = 20,
                                 current_weights: np.ndarray = None) -> np.ndarray:
    """Rolling std of equal-weight portfolio returns. 1 dim.

    Captures the overall risk level of the portfolio.
    """
    all_returns = []
    for ticker in raw_states:
        closes = _extract_closes(raw_states[ticker])
        returns = np.diff(closes) / (closes[:-1] + 1e-10)
        all_returns.append(returns)

    min_len = min(len(r) for r in all_returns)
    if min_len < 2:
        return np.array([0.0])

    aligned = np.array([r[-min_len:] for r in all_returns])
    port_returns = np.mean(aligned, axis=0)
    seg = port_returns[-window:] if len(port_returns) >= window else port_returns
    return np.array([float(np.std(seg, ddof=1))])


def compute_return_dispersion(raw_states: dict, window: int = 20,
                              current_weights: np.ndarray = None) -> np.ndarray:
    """Cross-sectional std of individual stock returns. 1 dim.

    High dispersion indicates strong divergence across stocks, which favours
    stock-picking strategies.
    """
    recent_rets = []
    for ticker in raw_states:
        closes = _extract_closes(raw_states[ticker])
        if len(closes) < 2:
            recent_rets.append(0.0)
            continue
        ret = (closes[-1] - closes[-2]) / (closes[-2] + 1e-10)
        recent_rets.append(ret)
    return np.array([float(np.std(recent_rets, ddof=1))])


def compute_sector_exposure(raw_states: dict, current_weights: np.ndarray = None,
                            growth_idx=None, defensive_idx=None) -> np.ndarray:
    """``[growth_weight, defensive_weight]`` from config-declared ticker groups. 2 dims."""
    if current_weights is None:
        return np.array([0.8, 0.2])
    g = growth_idx or []
    d = defensive_idx or []
    growth_weight = float(sum(current_weights[i] for i in g if i < len(current_weights)))
    defensive_weight = float(sum(current_weights[i] for i in d if i < len(current_weights)))
    return np.array([growth_weight, defensive_weight])


def compute_volume_breadth(raw_states: dict, window: int = 10,
                           current_weights: np.ndarray = None) -> np.ndarray:
    """Fraction of stocks with above-average volume. 1 dim.

    High breadth = broad market activity; low breadth = only a few stocks
    are receiving attention.
    """
    all_vol_ratios = []
    for ticker in raw_states:
        vols = _extract_volumes(raw_states[ticker])
        if len(vols) < window + 1:
            all_vol_ratios.append(1.0)
            continue
        avg_vol = np.mean(vols[-window - 1:-1]) + 1e-10
        ratio = vols[-1] / avg_vol
        all_vol_ratios.append(ratio)
    above_avg = sum(1 for r in all_vol_ratios if r > 1.0)
    return np.array([float(above_avg / len(raw_states))])


def compute_mean_reversion_score(raw_states: dict, window: int = 20,
                                 current_weights: np.ndarray = None) -> np.ndarray:
    """Z-score of each stock's current price vs its N-day mean. n dims.

    Positive z = overbought (potential pullback); negative z = oversold
    (potential rebound).
    """
    scores = []
    for ticker in raw_states:
        closes = _extract_closes(raw_states[ticker])
        if len(closes) < window:
            scores.append(0.0)
            continue
        seg = closes[-window:]
        mean_val = np.mean(seg)
        std_val = np.std(seg, ddof=1) + 1e-10
        z = (closes[-1] - mean_val) / std_val
        scores.append(float(np.clip(z, -3, 3)))
    return np.array(scores)


PORTFOLIO_INDICATOR_REGISTRY = {
    'momentum_rank': {
        'fn': compute_momentum_rank, 'output_dim': 5,
        'default_params': {'window': 20}, 'param_ranges': {'window': (10, 60)},
    },
    'rolling_correlation': {
        'fn': compute_rolling_correlation, 'output_dim': 10,
        'default_params': {'window': 60}, 'param_ranges': {'window': (20, 120)},
    },
    'relative_strength': {
        'fn': compute_relative_strength, 'output_dim': 5,
        'default_params': {'window': 20}, 'param_ranges': {'window': (10, 60)},
    },
    'portfolio_volatility': {
        'fn': compute_portfolio_volatility, 'output_dim': 1,
        'default_params': {'window': 20}, 'param_ranges': {'window': (10, 60)},
    },
    'return_dispersion': {
        'fn': compute_return_dispersion, 'output_dim': 1,
        'default_params': {'window': 20}, 'param_ranges': {'window': (10, 60)},
    },
    'sector_exposure': {
        'fn': compute_sector_exposure, 'output_dim': 2,
        'default_params': {}, 'param_ranges': {},
    },
    'volume_breadth': {
        'fn': compute_volume_breadth, 'output_dim': 1,
        'default_params': {'window': 10}, 'param_ranges': {'window': (5, 20)},
    },
    'mean_reversion_score': {
        'fn': compute_mean_reversion_score, 'output_dim': 5,
        'default_params': {'window': 20}, 'param_ranges': {'window': (10, 60)},
    },
}


def build_portfolio_features(selection: list, tickers: list = None,
                             growth: list = None, defensive: list = None) -> Callable:
    """Build a closure that computes all selected portfolio features.

    Combines the chosen portfolio-level indicators into a single callable;
    parameters are clipped to the ranges declared in
    ``PORTFOLIO_INDICATOR_REGISTRY``.

    Parameters
    ----------
    selection:
        List of indicator dicts with ``indicator`` and optional ``params`` keys.
    tickers:
        Ordered list of ticker symbols for the panel (insertion order matches
        the weight vector).
    growth:
        Tickers belonging to the growth group (used by ``sector_exposure``).
    defensive:
        Tickers belonging to the defensive group (used by ``sector_exposure``).
    """
    tickers = list(tickers) if tickers else []
    growth_idx = [tickers.index(t) for t in (growth or []) if t in tickers]
    defensive_idx = [tickers.index(t) for t in (defensive or []) if t in tickers]

    funcs = []
    output_dims = []

    for item in selection:
        name = item.get('indicator', '')
        params = dict(item.get('params', {}))
        if name not in PORTFOLIO_INDICATOR_REGISTRY:
            continue
        entry = PORTFOLIO_INDICATOR_REGISTRY[name]
        merged = dict(entry['default_params'])
        merged.update(params)
        if name == 'sector_exposure':
            merged['growth_idx'] = growth_idx
            merged['defensive_idx'] = defensive_idx
        for pk, pv in merged.items():
            if pk in entry['param_ranges']:
                lo, hi = entry['param_ranges'][pk]
                merged[pk] = type(pv)(np.clip(pv, lo, hi))
        funcs.append((entry['fn'], merged))
        output_dims.append(entry['output_dim'])

    if not funcs:
        def fallback(raw_states, current_weights=None):
            return np.zeros(3)
        return fallback

    _funcs = funcs
    _output_dims = output_dims

    def compute_portfolio_feats(raw_states, current_weights=None):
        features = []
        for idx, (fn, params) in enumerate(_funcs):
            try:
                result = fn(raw_states, current_weights=current_weights, **params)
                if not isinstance(result, np.ndarray):
                    result = np.atleast_1d(np.array(result, dtype=float))
                if result.ndim != 1:
                    result = result.flatten()
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    result = np.zeros(_output_dims[idx])
                features.append(result)
            except Exception:
                features.append(np.zeros(_output_dims[idx]))
        if not features:
            return np.zeros(3)
        return np.concatenate(features)

    return compute_portfolio_feats
