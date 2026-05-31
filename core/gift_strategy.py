"""
GIFT strategy for backtesting.

Loads a trained GIFT model and runs a backtest on the test period.
Used to validate the effect of the iterative optimization.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from feature_library import build_revise_state
from portfolio_features import build_portfolio_features
from reward_rules import build_reward_rules
from regime_detector import detect_market_regime
from portfolio_env import PortfolioEnv
from ppo_agent import PPOAgent
from metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio


def run_backtest(config: dict, model_path: str, config_path: str,
                 test_period: tuple = None,
                 transaction_cost: float = 0.001) -> dict:
    """Run backtest with a trained GIFT model.

    Args:
        config: full config dict
        model_path: path to saved model .pt file
        config_path: path to iteration config.json with feature/reward selections
        test_period: (start_date, end_date) for backtest
        transaction_cost: transaction cost rate

    Returns:
        dict with backtest metrics.

    Loads the iteration config (feature selections + reward rules), builds the
    environment, loads model weights, and runs a deterministic policy over the
    full test period while computing performance metrics.
    """
    # Load iteration config
    with open(config_path, 'r') as f:
        iter_config = json.load(f)

    # Build closures
    stock_features = iter_config.get('stock_features', [])
    portfolio_feats = iter_config.get('portfolio_features', [])
    reward_rules = iter_config.get('reward_rules', [])

    revise_fn = build_revise_state(stock_features) if stock_features else None
    reward_fn = build_reward_rules(reward_rules) if reward_rules else None

    # Create test environment
    data_cfg = config.get('data', {})
    tickers = list(data_cfg.get('tickers', []))
    growth = list(data_cfg.get('growth', []))
    defensive = list(data_cfg.get('defensive', []))
    port_feat_fn = build_portfolio_features(portfolio_feats, tickers=tickers,
                                            growth=growth, defensive=defensive) if portfolio_feats else None
    if test_period is None:
        test_period = tuple(config.get('experiment', {}).get(
            'test_period', ['2023-01-01', '2023-12-31']))

    env = PortfolioEnv(
        data_cfg.get('pickle_file', 'data/portfolio_5stocks.pkl'),
        config,
        revise_state_fn=revise_fn,
        portfolio_features_fn=port_feat_fn,
        reward_rules_fn=reward_fn,
        detect_regime_fn=detect_market_regime,
        train_period=test_period,
        transaction_cost=transaction_cost,
    )

    # Create agent and load weights
    state_dim = env.state_dim
    hidden_dim = config.get('ppo', {}).get('hidden_dim', 256)
    agent = PPOAgent(state_dim=state_dim, hidden_dim=hidden_dim)
    agent.load(model_path)

    # Run backtest
    state = env.reset()
    done = False
    returns = []
    weights_history = []
    portfolio_values = [1.0]

    while not done:
        weights, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(weights)
        returns.append(info.get('portfolio_return', 0))
        weights_history.append(info.get('weights', np.ones(6)/6).copy())
        portfolio_values.append(env.portfolio_value)
        state = next_state

    # Compute metrics
    bt_sharpe = sharpe_ratio(returns)
    bt_sortino = sortino_ratio(returns)
    bt_mdd = max_drawdown(returns)
    bt_calmar = calmar_ratio(returns)
    bt_total_return = (portfolio_values[-1] - 1.0) * 100

    # Weight statistics
    weights_arr = np.array(weights_history)
    avg_weights = weights_arr.mean(axis=0)
    turnover = np.mean(np.sum(np.abs(np.diff(weights_arr, axis=0)), axis=1)) / 2 if len(weights_arr) > 1 else 0

    result = {
        'sharpe': bt_sharpe,
        'sortino': bt_sortino,
        'max_drawdown': bt_mdd,
        'calmar': bt_calmar,
        'total_return': bt_total_return,
        'avg_turnover': float(turnover),
        'avg_weights': {t: float(avg_weights[i]) for i, t in enumerate(tickers)},
        'avg_cash_weight': float(avg_weights[len(tickers)]),
        'n_trading_days': len(returns),
        'final_value': float(portfolio_values[-1]),
    }

    print("\n" + "=" * 50)
    print("Backtest Results")
    print("=" * 50)
    print(f"Period: {test_period[0]} to {test_period[1]}")
    print(f"Total Return: {bt_total_return:.2f}%")
    print(f"Sharpe Ratio: {bt_sharpe:.3f}")
    print(f"Sortino Ratio: {bt_sortino:.3f}")
    print(f"Max Drawdown: {bt_mdd:.2f}%")
    print(f"Calmar Ratio: {bt_calmar:.3f}")
    print(f"Avg Turnover: {turnover:.4f}")
    print(f"\nAvg Weights:")
    for i, t in enumerate(tickers):
        print(f"  {t}: {avg_weights[i]:.3f}")
    print(f"  CASH: {avg_weights[len(tickers)]:.3f}")

    return result
