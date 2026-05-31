"""
Integration Test for Portfolio Optimization GIFT Pipeline

Tests each component in isolation, then runs a mini training loop.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Setup path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'core'))

def test_imports():
    """Test all modules can be imported."""
    print("1. Testing imports...")
    from metrics import sharpe_ratio, ic
    from feature_library import INDICATOR_REGISTRY, build_revise_state
    from portfolio_features import PORTFOLIO_INDICATOR_REGISTRY, build_portfolio_features
    from reward_rules import REWARD_RULE_REGISTRY, build_reward_rules
    from regime_detector import detect_market_regime
    from market_stats import get_market_stats
    from prompts import _extract_json
    print(f"   Stock indicators: {len(INDICATOR_REGISTRY)}")
    print(f"   Portfolio indicators: {len(PORTFOLIO_INDICATOR_REGISTRY)}")
    print(f"   Reward rules: {len(REWARD_RULE_REGISTRY)}")
    print("   OK")


def test_feature_library():
    """Test feature computation on synthetic data."""
    print("\n2. Testing feature_library...")
    from feature_library import build_revise_state, INDICATOR_REGISTRY

    # Synthetic 120d state (20 days * 6 channels)
    np.random.seed(42)
    raw_state = np.random.randn(120) * 10 + 100
    raw_state = np.abs(raw_state)

    # Test each indicator
    for name, entry in INDICATOR_REGISTRY.items():
        try:
            result = entry['fn'](raw_state, **entry['default_params'])
            assert isinstance(result, np.ndarray), f"{name}: not ndarray"
            assert result.shape == (entry['output_dim'],), \
                f"{name}: shape {result.shape} != ({entry['output_dim']},)"
            assert not np.any(np.isnan(result)), f"{name}: NaN"
            assert not np.any(np.isinf(result)), f"{name}: Inf"
        except Exception as e:
            print(f"   FAIL {name}: {e}")
            return False

    # Test closure
    selection = [
        {'indicator': 'RSI', 'params': {'window': 14}},
        {'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
    ]
    fn = build_revise_state(selection)
    result = fn(raw_state)
    assert isinstance(result, np.ndarray), "closure not ndarray"
    assert len(result) == 4, f"closure dim {len(result)} != 4"
    print(f"   All {len(INDICATOR_REGISTRY)} indicators OK, closure dim={len(result)}")
    print("   OK")
    return True


def test_portfolio_features():
    """Test portfolio-level features."""
    print("\n3. Testing portfolio_features...")
    from portfolio_features import build_portfolio_features, PORTFOLIO_INDICATOR_REGISTRY
    TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']

    np.random.seed(42)
    raw_states = {}
    for t in TICKERS:
        raw_states[t] = np.abs(np.random.randn(120) * 10 + 100)

    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    selection = [
        {'indicator': 'momentum_rank', 'params': {'window': 20}},
        {'indicator': 'portfolio_volatility', 'params': {'window': 20}},
    ]
    fn = build_portfolio_features(selection)
    result = fn(raw_states, current_weights=weights)
    assert isinstance(result, np.ndarray), "not ndarray"
    assert not np.any(np.isnan(result)), "NaN in output"
    print(f"   Output dim: {len(result)}")
    print("   OK")
    return True


def test_reward_rules():
    """Test reward rules."""
    print("\n4. Testing reward_rules...")
    from reward_rules import build_reward_rules

    selection = [
        {'rule': 'penalize_concentration', 'params': {'max_weight': 0.35}},
        {'rule': 'reward_diversification', 'params': {'min_stocks': 3}},
    ]
    fn = build_reward_rules(selection)

    # Test with concentrated weights
    weights = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    bonus, log = fn(weights=weights)
    assert isinstance(bonus, float), "bonus not float"
    assert isinstance(log, dict), "log not dict"
    print(f"   Concentrated weights: bonus={bonus:.4f}, triggers={log}")
    print("   OK")
    return True


def test_regime_detector():
    """Test regime detector."""
    print("\n5. Testing regime_detector...")
    from regime_detector import detect_market_regime
    TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']

    np.random.seed(42)
    raw_states = {}
    for t in TICKERS:
        raw_states[t] = np.abs(np.random.randn(120) * 10 + 100)

    regime = detect_market_regime(raw_states)
    assert len(regime) == 3, f"regime dim {len(regime)} != 3"
    assert -1 <= regime[0] <= 1, f"trend {regime[0]} out of range"
    assert 0 <= regime[1] <= 1, f"vol {regime[1]} out of range"
    assert 0 <= regime[2] <= 1, f"risk {regime[2]} out of range"
    print(f"   Regime: trend={regime[0]:.3f}, vol={regime[1]:.3f}, risk={regime[2]:.3f}")
    print("   OK")
    return True


def test_prompts():
    """Test prompt building and JSON extraction."""
    print("\n6. Testing prompts...")
    from prompts import _extract_json, build_init_prompt

    # Test JSON extraction
    json_text = '```json\n{"features": [{"indicator": "RSI", "params": {"window": 14}}]}\n```'
    parsed = _extract_json(json_text)
    assert 'features' in parsed, "no features key"
    assert parsed['features'][0]['indicator'] == 'RSI', "wrong indicator"

    # Test code-generation prompt building (current GIFT API)
    prompt = build_init_prompt("test market stats")
    assert isinstance(prompt, str) and len(prompt) > 500, "init prompt too short"
    assert 'TSLA' in prompt, "ticker missing from init prompt"
    assert 'revise_state' in prompt, "revise_state function name missing"
    assert 'intrinsic_reward' in prompt, "intrinsic_reward function name missing"

    print("   JSON extraction + prompt building OK")
    print("   OK")
    return True


def test_env_and_ppo():
    """Test environment + PPO agent (short training)."""
    print("\n7. Testing PortfolioEnv + PPOAgent...")
    import yaml
    from portfolio_env import PortfolioEnv
    from ppo_agent import PPOAgent
    from feature_library import build_revise_state
    from portfolio_features import build_portfolio_features
    from regime_detector import detect_market_regime

    # Load config
    config_path = str(PROJECT_DIR / 'configs' / 'config.yaml')
    if not os.path.exists(config_path):
        print("   SKIP: config.yaml not found")
        return True

    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_path = config['data']['pickle_file']
    if not os.path.exists(data_path):
        print(f"   SKIP: {data_path} not found")
        return True

    # Build simple features
    revise_fn = build_revise_state([
        {'indicator': 'RSI', 'params': {'window': 14}},
        {'indicator': 'Momentum', 'params': {'window': 10}},
    ])
    port_fn = build_portfolio_features([
        {'indicator': 'momentum_rank', 'params': {'window': 20}},
    ])

    # Create env
    env = PortfolioEnv(
        data_path, config,
        revise_state_fn=revise_fn,
        portfolio_features_fn=port_fn,
        detect_regime_fn=detect_market_regime,
        train_period=('2020-01-01', '2020-06-30'),  # short period
        transaction_cost=0.001,
    )

    state = env.reset()
    state_dim = len(state)
    print(f"   State dim: {state_dim}")

    # Create agent
    agent = PPOAgent(state_dim=state_dim, hidden_dim=64,
                     actor_lr=3e-4, critic_lr=1e-3)

    # Run 2 episodes
    for ep in range(2):
        state = env.reset()
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            weights, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(weights)
            states.append(state)
            actions.append(weights)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            state = next_state

        if len(states) > 1:
            agent.update(states, actions, log_probs, rewards, dones, state)

    print(f"   Ran 2 episodes successfully")
    print("   OK")
    return True


if __name__ == '__main__':
    print("=" * 50)
    print("Integration Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_feature_library,
        test_portfolio_features,
        test_reward_rules,
        test_regime_detector,
        test_prompts,
        test_env_and_ppo,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            result = test_fn()
            if result is not False:
                passed += 1
        except Exception as e:
            print(f"   FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
