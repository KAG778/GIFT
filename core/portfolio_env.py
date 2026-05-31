"""
Portfolio environment for RL training.

Simulates portfolio management with the configured stock panel (N stocks + cash).

State vector layout:
  - Compressed raw state: 10 dims * N stocks
  - ``revise_state`` extras:  K dims * N stocks
  - Portfolio-level features:  P dims
  - Market regime vector:      3 dims (trend / volatility / risk)
  - Current weights:           N+1 dims

Action: (N+1)-dim target weights sampled from a Dirichlet (always summing to 1).
Reward: base mean-variance term + LLM-selected reward rules + intrinsic
reward emitted by the LLM-generated code.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional

# Fallback only — the active list comes from config['data']['tickers'].
DEFAULT_TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']
WINDOW = 20   # lookback window in trading days
STATE_CHANNELS = 6  # close, open, high, low, volume, adj_close per day


class PortfolioEnv:
    """RL environment for portfolio optimization.

    Core API: ``reset()`` -> repeated ``step(weights)`` until ``done``.
    """

    def __init__(self, data_path: str, config: dict,
                 revise_state_fn: Callable = None,
                 portfolio_features_fn: Callable = None,
                 reward_rules_fn: Callable = None,
                 detect_regime_fn: Callable = None,
                 intrinsic_reward_fn: Callable = None,
                 train_period: tuple = None,
                 transaction_cost: float = 0.001):
        """Initialize environment.

        Args:
            data_path: path to pickle file from prepare_data.py
            config: config dict with keys from config.yaml
            revise_state_fn: closure from feature_library.build_revise_state()
            portfolio_features_fn: closure from portfolio_features.build_portfolio_features()
            reward_rules_fn: closure from reward_rules.build_reward_rules()
            detect_regime_fn: regime_detector.detect_market_regime
            train_period: (start_date, end_date) strings
            transaction_cost: transaction cost rate
        """
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.config = config
        self.tickers = list(config.get('data', {}).get('tickers', DEFAULT_TICKERS))
        self.n_assets = len(self.tickers) + 1  # stocks + cash
        self.revise_state_fn = revise_state_fn
        self.portfolio_features_fn = portfolio_features_fn
        self.reward_rules_fn = reward_rules_fn
        self.detect_regime_fn = detect_regime_fn
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.transaction_cost = transaction_cost

        # Sort dates
        self.all_dates = sorted(self.raw_data.keys())
        if train_period:
            self.dates = [d for d in self.all_dates
                          if train_period[0] <= d <= train_period[1]]
        else:
            self.dates = self.all_dates

        # Pre-extract price matrices
        self._build_price_matrix()

        # State tracking
        self.current_step = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.peak_value = 1.0

    def _build_price_matrix(self):
        """Build a date-aligned price + volume matrix for fast lookup."""
        self.prices = {}
        self.volumes = {}

        for date in self.dates:
            day_data = self.raw_data.get(date, {})
            price_row = {}
            vol_row = {}
            for ticker in self.tickers:
                info = day_data.get('price', {}).get(ticker, {})
                close = info.get('close', 0.0)
                adj_close = info.get('adjusted_close', close)
                vol = info.get('volume', 0.0)
                price_row[ticker] = adj_close if adj_close > 0 else close
                vol_row[ticker] = vol
            self.prices[date] = price_row
            self.volumes[date] = vol_row

    def _get_raw_state(self, ticker: str, date_idx: int) -> np.ndarray:
        """Build the 120-d interleaved raw state for a single ticker.

        Layout: ``[close, open, high, low, volume, adj_close] * 20 days``.
        """
        start_idx = max(0, date_idx - WINDOW + 1)
        end_idx = date_idx + 1

        state = np.zeros(WINDOW * STATE_CHANNELS)
        for i, didx in enumerate(range(start_idx, end_idx)):
            if didx < 0 or didx >= len(self.dates):
                continue
            date = self.dates[didx]
            day_data = self.raw_data.get(date, {}).get('price', {}).get(ticker, {})
            offset = i * STATE_CHANNELS
            state[offset + 0] = day_data.get('close', 0.0)
            state[offset + 1] = day_data.get('open', 0.0)
            state[offset + 2] = day_data.get('high', 0.0)
            state[offset + 3] = day_data.get('low', 0.0)
            state[offset + 4] = day_data.get('volume', 0.0)
            state[offset + 5] = day_data.get('adjusted_close',
                                              day_data.get('close', 0.0))
        return state

    def _get_raw_states_dict(self, date_idx: int) -> Dict[str, np.ndarray]:
        """Get raw states for all tickers."""
        return {t: self._get_raw_state(t, date_idx) for t in self.tickers}

    def _compress_raw_state(self, raw_state: np.ndarray) -> np.ndarray:
        """Compress the 120-dim raw state into ~10 dims.

        Keeps 5 normalised recent closes + 5 recent returns. This retains the
        most informative pieces while greatly shrinking the state dimension.
        """
        closes = np.array([raw_state[i * 6] for i in range(WINDOW)], dtype=float)
        mean_c = np.mean(closes) + 1e-10
        recent_closes = (closes[-5:] / mean_c) - 1.0
        returns = np.diff(closes) / (closes[:-1] + 1e-10)
        recent_returns = returns[-5:] if len(returns) >= 5 else np.zeros(5)
        return np.concatenate([recent_closes, recent_returns])

    def _compute_state(self, date_idx: int) -> np.ndarray:
        """Compute the full observation vector.

        Layout (concatenated):
          - Compressed raw state:        10 dims * 5 stocks = 50
          - Revised features (extras):    K dims * 5 stocks  (LLM-generated)
          - Portfolio-level features:     P dims
          - Regime vector:                 3 dims (trend / volatility / risk)
          - Current weights:               6 dims
        """
        raw_states = self._get_raw_states_dict(date_idx)

        parts = []

        # Compressed raw per stock (10 * N)
        compressed = []
        for ticker in self.tickers:
            compressed.append(self._compress_raw_state(raw_states[ticker]))
        parts.append(np.concatenate(compressed))

        # Revised extras per stock (K * 5)
        if self.revise_state_fn:
            # First pass: detect extras dimension from any stock that produces extras
            extras_dim = None
            for ticker in self.tickers:
                full_revised = self.revise_state_fn(raw_states[ticker])
                if len(full_revised) > 120:
                    extras_dim = len(full_revised) - 120
                    break
            # Second pass: build extras with consistent dimension
            revised_per_stock = []
            for ticker in self.tickers:
                full_revised = self.revise_state_fn(raw_states[ticker])
                if len(full_revised) > 120:
                    extras = full_revised[120:]
                else:
                    extras = np.zeros(extras_dim) if extras_dim is not None else np.array([])
                if np.any(np.isnan(extras)) or np.any(np.isinf(extras)):
                    extras = np.zeros_like(extras) if len(extras) > 0 else extras
                revised_per_stock.append(extras)
            if extras_dim is not None:
                parts.append(np.concatenate(revised_per_stock))
            else:
                parts.append(np.zeros(5))
        else:
            parts.append(np.zeros(5))

        # Portfolio features
        if self.portfolio_features_fn:
            port_feat = self.portfolio_features_fn(raw_states, self.weights)
            parts.append(port_feat)
        else:
            parts.append(np.zeros(5))

        # Regime
        if self.detect_regime_fn:
            regime = self.detect_regime_fn(raw_states)
        else:
            regime = np.array([0.0, 0.5, 0.0])
        parts.append(regime)

        # Current weights
        parts.append(self.weights)

        return np.concatenate(parts)

    @property
    def state_dim(self):
        """Compute the state dimension via a dry run.

        Needed because the dimension depends on the number of features the
        LLM-generated ``revise_state`` adds.
        """
        if not self.dates:
            return 100
        state = self._compute_state(0)
        return len(state)

    def reset(self) -> np.ndarray:
        """Reset the environment to the start of the period.

        Weights are initialised to equal-weight ``(1/6)`` and portfolio value
        to ``1.0``.
        """
        self.current_step = WINDOW  # need WINDOW days of history
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        return self._compute_state(self.current_step)

    def step(self, target_weights: np.ndarray):
        """Execute one trading step.

        Args:
            target_weights: array of shape ``(6,)``; should sum to 1.

        Returns:
            ``next_state, reward, done, info``.

        Pipeline: normalise target weights -> compute portfolio return with
        transaction cost -> reward = base mean-variance + reward rules +
        intrinsic reward -> emit next state and info dict.
        """
        # Normalize weights
        target_weights = np.clip(target_weights, 0, 1)
        total = target_weights.sum()
        if total > 1e-8:
            target_weights = target_weights / total
        else:
            target_weights = np.ones(self.n_assets) / self.n_assets

        prev_weights = self.weights.copy()
        self.current_step += 1

        if self.current_step >= len(self.dates):
            done = True
            return np.zeros(self.state_dim), 0.0, done, {}

        done = self.current_step >= len(self.dates) - 1

        # Compute portfolio return
        date = self.dates[self.current_step]
        prev_date = self.dates[self.current_step - 1]

        stock_returns = np.zeros(len(self.tickers))
        for i, ticker in enumerate(self.tickers):
            prev_price = self.prices.get(prev_date, {}).get(ticker, 0.0)
            curr_price = self.prices.get(date, {}).get(ticker, 0.0)
            if prev_price > 0:
                stock_returns[i] = (curr_price - prev_price) / prev_price
            else:
                stock_returns[i] = 0.0

        # Cash return (0 for simplicity, or risk-free rate)
        cash_return = 0.0
        all_returns = np.append(stock_returns, cash_return)

        # Portfolio return before transaction costs
        port_return = float(np.dot(prev_weights, all_returns))

        # Transaction cost
        turnover = float(np.sum(np.abs(target_weights - prev_weights))) / 2.0
        tc_cost = turnover * self.transaction_cost

        # Net return
        net_return = port_return - tc_cost

        # Update portfolio value
        self.portfolio_value *= (1 + net_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Update weights (drift + rebalance)
        drifted = prev_weights * (1 + all_returns)
        drifted_sum = drifted.sum()
        if drifted_sum > 1e-8:
            drifted = drifted / drifted_sum
        self.weights = target_weights

        # Compute reward
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Base reward: mean-variance
        lambda_mv = self.config.get('portfolio', {}).get('default_lambda', 0.5)
        base_reward = net_return - lambda_mv * (current_drawdown ** 2)

        # Additional reward rules
        rule_bonus = 0.0
        trigger_log = {}
        if self.reward_rules_fn:
            raw_states = self._get_raw_states_dict(self.current_step)
            regime = self.detect_regime_fn(raw_states) if self.detect_regime_fn else None
            port_feats = {}
            if self.portfolio_features_fn:
                pf_raw = self.portfolio_features_fn(raw_states, self.weights)
                port_feats['raw'] = pf_raw

            # Compute momentum_rank for momentum_alignment rule
            try:
                from core.portfolio_features import compute_momentum_rank
                port_feats['momentum_rank'] = compute_momentum_rank(raw_states, current_weights=self.weights)
            except Exception:
                pass

            rule_bonus, trigger_log = self.reward_rules_fn(
                weights=self.weights,
                prev_weights=prev_weights,
                regime_vector=regime,
                portfolio_features=port_feats,
                base_reward=base_reward,
                current_drawdown=current_drawdown,
            )

        # Intrinsic reward from LLM code (averaged across all tickers)
        intrinsic_r = 0.0
        if self.intrinsic_reward_fn and self.revise_state_fn:
            try:
                raw_states_now = self._get_raw_states_dict(self.current_step)
                ir_values = []
                for ticker in self.tickers:
                    revised = self.revise_state_fn(raw_states_now[ticker])
                    ir_values.append(float(self.intrinsic_reward_fn(revised)))
                intrinsic_r = float(np.mean(ir_values))
                intrinsic_r = np.clip(intrinsic_r, -1.0, 1.0)
            except Exception:
                intrinsic_r = 0.0

        reward = base_reward + rule_bonus + intrinsic_r

        # Next state
        next_state = self._compute_state(self.current_step)

        info = {
            'portfolio_return': net_return,
            'transaction_cost': tc_cost,
            'turnover': turnover,
            'drawdown': current_drawdown,
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'trigger_log': trigger_log,
            'intrinsic_reward': float(intrinsic_r),
        }

        return next_state, reward, done, info

    def get_revised_states(self, n_samples: int = 200) -> dict:
        """Per-ticker revised states + forward returns for IC computation.

        Returns a dict with:
          - ``'revised_states_per_ticker'``: ``{ticker: (N, state_dim)}``
          - ``'forward_returns'``:           ``(N,)``
          - ``'regime_labels'``:             ``(N,)``
        """
        if self.revise_state_fn is None:
            return {
                'revised_states_per_ticker': {},
                'forward_returns': np.array([]),
                'regime_labels': np.array([]),
            }

        n = min(n_samples, len(self.dates) - WINDOW - 1)
        if n < 10:
            return {
                'revised_states_per_ticker': {},
                'forward_returns': np.array([]),
                'regime_labels': np.array([]),
            }
        indices = np.linspace(WINDOW, len(self.dates) - 2, n, dtype=int)

        revised_per_ticker = {t: [] for t in self.tickers}
        forward_list = []
        regime_labels = []

        for idx in indices:
            raw_states = self._get_raw_states_dict(idx)
            for ticker in self.tickers:
                revised = self.revise_state_fn(raw_states[ticker])
                revised_per_ticker[ticker].append(revised)

            date = self.dates[idx]
            next_date = self.dates[idx + 1]
            ret = 0.0
            for ticker in self.tickers:
                p0 = self.prices.get(date, {}).get(ticker, 0.0)
                p1 = self.prices.get(next_date, {}).get(ticker, 0.0)
                if p0 > 0:
                    ret += (p1 - p0) / p0
            forward_list.append(ret / len(self.tickers))

            if self.detect_regime_fn:
                rv = self.detect_regime_fn(raw_states)
                from ic_analyzer import _classify_regime
                regime_labels.append(_classify_regime(rv[0], rv[1]))
            else:
                regime_labels.append('neutral')

        revised_states_per_ticker = {
            t: np.array(revised_per_ticker[t]) for t in self.tickers
        }
        return {
            'revised_states_per_ticker': revised_states_per_ticker,
            'forward_returns': np.array(forward_list),
            'regime_labels': np.array(regime_labels),
        }

    def get_training_states(self, n_samples: int = 200) -> tuple:
        """Sample states + forward returns for feature screening / prompt stats.

        Returns:
            ``training_states``: dict ``{ticker: array_of_states}``.
            ``forward_returns``: forward returns of the equal-weight portfolio.
        """
        n = min(n_samples, len(self.dates) - WINDOW - 1)
        indices = np.linspace(WINDOW, len(self.dates) - 2, n, dtype=int)

        training_states = {t: [] for t in self.tickers}
        forward_returns = []

        for idx in indices:
            for ticker in self.tickers:
                training_states[ticker].append(self._get_raw_state(ticker, idx))

            # Forward return: equal-weight portfolio
            date = self.dates[idx]
            next_date = self.dates[idx + 1]
            ret = 0.0
            for ticker in self.tickers:
                p0 = self.prices.get(date, {}).get(ticker, 0.0)
                p1 = self.prices.get(next_date, {}).get(ticker, 0.0)
                if p0 > 0:
                    ret += (p1 - p0) / p0
            forward_returns.append(ret / len(self.tickers))

        for ticker in self.tickers:
            training_states[ticker] = np.array(training_states[ticker])
        forward_returns = np.array(forward_returns)

        return training_states, forward_returns
