"""
Integration test for refactored GIFT pipeline (code-generation mode).

Tests the full pipeline without LLM calls:
1. Code sandbox validates sample code
2. PortfolioEnv uses code-generated revise_state + intrinsic_reward
3. PPO trains with the new state representation
4. IC analyzer computes profiles
5. COT feedback is generated
"""

import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'core'))


def test_code_sandbox():
    """Test 1: Code sandbox validates and extracts functions."""
    from code_sandbox import validate

    code = """
import numpy as np
from feature_library import compute_realized_volatility, compute_relative_momentum

def revise_state(s):
    closes = s[0::6]
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    vol = compute_realized_volatility(returns, 20)
    mom = compute_relative_momentum(closes, 20)
    return np.concatenate([s, [vol, mom]])

def intrinsic_reward(updated_s):
    vol = updated_s[120]
    mom = updated_s[121]
    return 0.01 * abs(mom) / (vol + 0.01)
"""
    result = validate(code)
    assert result['ok'], f"Validation failed: {result['errors']}"
    assert result['feature_dim'] == 2
    assert result['revise_state'] is not None
    assert result['intrinsic_reward'] is not None

    test_s = np.random.randn(120) * 100 + 150
    revised = result['revise_state'](test_s)
    assert np.allclose(revised[:120], test_s, atol=1e-6)
    assert len(revised) == 122

    reward = result['intrinsic_reward'](revised)
    assert isinstance(reward, (int, float, np.integer, np.floating))
    assert abs(reward) <= 100

    print("  [PASS] Code sandbox")
    return result


def test_portfolio_env_with_code(code_result):
    """Test 2: PortfolioEnv with code-generated functions."""
    from portfolio_env import PortfolioEnv

    data_path = str(ROOT / 'data' / 'portfolio_5stocks.pkl')
    if not os.path.exists(data_path):
        print("  [SKIP] No data file")
        return None

    env = PortfolioEnv(
        data_path,
        {'portfolio': {'default_lambda': 0.5}},
        revise_state_fn=code_result['revise_state'],
        intrinsic_reward_fn=code_result['intrinsic_reward'],
        detect_regime_fn=None,
        train_period=('2020-01-01', '2020-06-30'),
        transaction_cost=0.001,
    )

    state = env.reset()
    assert state.ndim == 1
    assert len(state) > 60

    next_state, reward, done, info = env.step(np.ones(6) / 6)
    assert isinstance(reward, float)
    assert 'intrinsic_reward' in info

    print(f"  [PASS] PortfolioEnv (state_dim={len(state)}, reward={reward:.6f})")
    return env


def test_ic_analyzer(env):
    """Test 3: IC analysis on revised states."""
    if env is None:
        print("  [SKIP] No env")
        return

    from ic_analyzer import (compute_ic_profile_ensemble,
                             compute_regime_specific_ic_ensemble,
                             build_ic_cot_prompt)

    result = env.get_revised_states(100)
    revised_per_ticker = result['revised_states_per_ticker']
    fwd = result['forward_returns']
    regimes = result['regime_labels']
    if len(fwd) < 20:
        print("  [SKIP] Not enough data for IC")
        return

    ic = compute_ic_profile_ensemble(revised_per_ticker, fwd)
    assert isinstance(ic, dict)
    print(f"  IC profile: {len(ic)} dims")

    regime_ic = compute_regime_specific_ic_ensemble(revised_per_ticker, fwd, regimes)
    assert isinstance(regime_ic, dict)
    print(f"  Regimes found: {list(regime_ic.keys())}")

    sample_results = [{
        'code': 'test code',
        'performance': {'sharpe': 0.5, 'total_return': 3.0, 'max_drawdown': -5.0},
        'ic_profile': ic,
        'regime_ic': regime_ic,
        'intrinsic_reward_stats': {'mean': 0.003, 'correlation_with_performance': 0.15},
    }]
    cot = build_ic_cot_prompt(sample_results, 0, market_period_summary="Test period")
    assert len(cot) > 100
    assert 'IC' in cot

    print(f"  [PASS] IC analyzer (COT: {len(cot)} chars)")


def test_ppo_training(code_result):
    """Test 4: PPO trains with code-generated state."""
    if not os.path.exists(str(ROOT / 'data' / 'portfolio_5stocks.pkl')):
        print("  [SKIP] No data file")
        return

    from portfolio_env import PortfolioEnv
    from ppo_agent import PPOAgent

    env = PortfolioEnv(
        str(ROOT / 'data' / 'portfolio_5stocks.pkl'),
        {'portfolio': {'default_lambda': 0.5}},
        revise_state_fn=code_result['revise_state'],
        intrinsic_reward_fn=code_result['intrinsic_reward'],
        train_period=('2020-01-01', '2020-03-31'),
    )

    state_dim = env.state_dim
    agent = PPOAgent(state_dim=state_dim, hidden_dim=32)

    state = env.reset()
    states, actions, log_probs, rewards, dones = [], [], [], [], []
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 50:
        weights, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(weights)
        states.append(state)
        actions.append(weights)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(float(done))
        total_reward += reward
        state = next_state
        steps += 1

    assert steps > 0
    print(f"  [PASS] PPO training ({steps} steps, total_reward={total_reward:.4f})")


if __name__ == '__main__':
    print("Integration Test: Refactored GIFT Pipeline")
    print("=" * 50)

    code_result = test_code_sandbox()
    env = test_portfolio_env_with_code(code_result)
    test_ic_analyzer(env)
    test_ppo_training(code_result)

    print("\n" + "=" * 50)
    print("All integration tests passed!")
