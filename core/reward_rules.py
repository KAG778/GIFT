"""
Reward rules for portfolio optimization.

Seven predefined reward rules that shape PPO behaviour on top of the base
mean-variance reward. The LLM selects and parameterizes them on each iteration
(emitted as JSON):

1. ``penalize_concentration`` — penalize over-concentration in a single stock.
2. ``reward_diversification`` — bonus for holding several stocks.
3. ``penalize_turnover``      — penalize excessive trading.
4. ``regime_defensive``       — reward holding cash in high-risk regimes.
5. ``momentum_alignment``     — bonus when weights align with momentum rank.
6. ``volatility_scaling``     — scale the reward down in high-vol regimes.
7. ``drawdown_penalty``       — penalize drawdowns beyond a threshold.
"""

import numpy as np
from typing import Callable


def rule_penalize_concentration(weights: np.ndarray, params: dict,
                                regime_vector: np.ndarray = None, **kwargs) -> float:
    """Penalize over-concentration: penalty applied when any stock weight exceeds ``max_weight``."""
    max_weight = params.get('max_weight', 0.35)
    penalty = params.get('penalty', 0.1)
    stock_weights = weights[:5]
    max_w = float(np.max(stock_weights))
    if max_w > max_weight:
        return -penalty * (max_w - max_weight) / (1.0 - max_weight + 1e-8)
    return 0.0


def rule_reward_diversification(weights: np.ndarray, params: dict,
                                regime_vector: np.ndarray = None, **kwargs) -> float:
    """Reward diversification: bonus when at least ``min_stocks`` stocks each have weight > 5%."""
    min_stocks = int(params.get('min_stocks', 3))
    bonus = params.get('bonus', 0.05)
    stock_weights = weights[:5]
    held = int(np.sum(stock_weights > 0.05))
    if held >= min_stocks:
        return bonus
    return 0.0


def rule_penalize_turnover(weights: np.ndarray, params: dict,
                           prev_weights: np.ndarray = None, **kwargs) -> float:
    """Penalize turnover: penalty applied when weight change exceeds ``threshold``."""
    threshold = params.get('threshold', 0.1)
    penalty = params.get('penalty', 0.15)
    if prev_weights is None:
        return 0.0
    turnover = float(np.sum(np.abs(weights - prev_weights))) / 2.0
    if turnover > threshold:
        return -penalty * (turnover - threshold)
    return 0.0


def rule_regime_defensive(weights: np.ndarray, params: dict,
                          regime_vector: np.ndarray = None, **kwargs) -> float:
    """Defensive-cash bonus: reward holding > 20% cash when risk_level is elevated."""
    crisis_threshold = params.get('crisis_threshold', 0.6)
    cash_bonus = params.get('cash_bonus', 0.1)
    if regime_vector is None:
        return 0.0
    risk_level = float(regime_vector[2])
    cash_weight = float(weights[5]) if len(weights) > 5 else 0.0
    if risk_level > crisis_threshold and cash_weight > 0.2:
        return cash_bonus * cash_weight
    return 0.0


def rule_momentum_alignment(weights: np.ndarray, params: dict,
                            portfolio_features: dict = None, **kwargs) -> float:
    """Momentum alignment: bonus when current weights correlate positively with the momentum rank."""
    bonus = params.get('bonus', 0.05)
    if portfolio_features is None or 'momentum_rank' not in portfolio_features:
        return 0.0
    ranks = portfolio_features['momentum_rank']
    stock_weights = weights[:5]
    if np.std(stock_weights) < 1e-8 or np.std(ranks) < 1e-8:
        return 0.0
    corr = np.corrcoef(stock_weights, ranks)[0, 1]
    if np.isnan(corr):
        return 0.0
    if corr > 0:
        return bonus * corr
    return 0.0


def rule_volatility_scaling(weights: np.ndarray, params: dict,
                            regime_vector: np.ndarray = None,
                            base_reward: float = 0.0, **kwargs) -> float:
    """Volatility scaling: scale the base reward down in high-volatility regimes to dampen aggressive strategies."""
    vol_threshold = params.get('vol_threshold', 0.5)
    scale = params.get('scale', 0.5)
    if regime_vector is None:
        return 0.0
    vol_level = float(regime_vector[1])
    if vol_level > vol_threshold:
        return -abs(base_reward) * (1.0 - scale) * (vol_level - vol_threshold) / (1.0 - vol_threshold + 1e-8)
    return 0.0


def rule_drawdown_penalty(weights: np.ndarray, params: dict,
                          current_drawdown: float = 0.0, **kwargs) -> float:
    """Drawdown penalty: penalty applied when the current drawdown exceeds ``dd_threshold``."""
    dd_threshold = params.get('dd_threshold', 0.1)
    penalty = params.get('penalty', 0.15)
    if current_drawdown > dd_threshold:
        return -penalty * (current_drawdown - dd_threshold) / (1.0 - dd_threshold + 1e-8)
    return 0.0


REWARD_RULE_REGISTRY = {
    'penalize_concentration': {
        'fn': rule_penalize_concentration,
        'default_params': {'max_weight': 0.35, 'penalty': 0.1},
        'param_ranges': {'max_weight': (0.2, 0.5), 'penalty': (0.01, 0.2)},
        'description': 'Penalty when any weight > max_weight',
    },
    'reward_diversification': {
        'fn': rule_reward_diversification,
        'default_params': {'min_stocks': 3, 'bonus': 0.05},
        'param_ranges': {'min_stocks': (2, 5), 'bonus': (0.01, 0.1)},
        'description': 'Bonus when holding >= min_stocks above 5%',
    },
    'penalize_turnover': {
        'fn': rule_penalize_turnover,
        'default_params': {'threshold': 0.1, 'penalty': 0.15},
        'param_ranges': {'threshold': (0.05, 0.3), 'penalty': (0.01, 0.2)},
        'description': 'Penalty when turnover > threshold',
    },
    'regime_defensive': {
        'fn': rule_regime_defensive,
        'default_params': {'crisis_threshold': 0.6, 'cash_bonus': 0.1},
        'param_ranges': {'crisis_threshold': (0.5, 0.8), 'cash_bonus': (0.01, 0.15)},
        'description': 'Bonus for high cash in high-risk regime',
    },
    'momentum_alignment': {
        'fn': rule_momentum_alignment,
        'default_params': {'bonus': 0.05},
        'param_ranges': {'bonus': (0.01, 0.1)},
        'description': 'Bonus when weights align with momentum rank',
    },
    'volatility_scaling': {
        'fn': rule_volatility_scaling,
        'default_params': {'vol_threshold': 0.5, 'scale': 0.5},
        'param_ranges': {'vol_threshold': (0.3, 0.8), 'scale': (0.3, 0.8)},
        'description': 'Scale down reward in high-vol regime',
    },
    'drawdown_penalty': {
        'fn': rule_drawdown_penalty,
        'default_params': {'dd_threshold': 0.1, 'penalty': 0.15},
        'param_ranges': {'dd_threshold': (0.05, 0.2), 'penalty': (0.05, 0.3)},
        'description': 'Penalty when drawdown exceeds threshold',
    },
}


def build_reward_rules(selection: list) -> Callable:
    """Build a closure that aggregates all selected reward rules.

    Combines the LLM-selected rules into a single callable; per-parameter
    values are clipped to the ranges declared in ``REWARD_RULE_REGISTRY``.
    """
    rule_funcs = []
    for item in selection:
        name = item.get('rule', '')
        params = dict(item.get('params', {}))
        if name not in REWARD_RULE_REGISTRY:
            continue
        entry = REWARD_RULE_REGISTRY[name]
        merged = dict(entry['default_params'])
        merged.update(params)
        for pk, pv in merged.items():
            if pk in entry['param_ranges']:
                lo, hi = entry['param_ranges'][pk]
                merged[pk] = type(pv)(np.clip(pv, lo, hi))
        rule_funcs.append((entry['fn'], merged, name))

    if not rule_funcs:
        def no_rules(**kwargs):
            return 0.0, {}
        return no_rules

    _rule_funcs = rule_funcs

    def compute_rules(**kwargs):
        total = 0.0
        trigger_log = {}
        for fn, params, name in _rule_funcs:
            val = fn(params=params, **kwargs)
            total += val
            trigger_log[name] = {
                'value': float(val),
                'triggered': abs(val) > 1e-6,
            }
        return float(total), trigger_log

    return compute_rules
