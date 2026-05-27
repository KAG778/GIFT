"""Run training for a single window, save results to JSON."""
import sys
import os
import json
import time
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'core'))
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

import yaml
import torch

from code_sandbox import validate as sandbox_validate
from feature_library import build_revise_state
from portfolio_features import build_portfolio_features
from regime_detector import detect_market_regime
from portfolio_env import PortfolioEnv
from ppo_agent import PPOAgent, set_seed
from metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio
from reward_rules import REWARD_RULE_REGISTRY, build_reward_rules

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']
MAX_EPISODES = 50


def load_best_gift_code(window, seed=123):
    results_base = PROJECT_DIR / 'results'
    result_dir = results_base / f'{window}_seed_{seed}_rerun'
    summary_path = result_dir / 'summary.json'
    if not summary_path.exists():
        return None, None
    with open(summary_path) as f:
        summary = json.load(f)
    best_iter = summary.get('best_iteration')
    if best_iter is None:
        return None, None
    iter_dir = result_dir / f'iteration_{best_iter}'
    code_path = iter_dir / 'code.py'
    config_path = iter_dir / 'config.json'
    if not code_path.exists():
        return None, None
    with open(code_path) as f:
        code = f.read()
    result = sandbox_validate(code)
    if not result['ok']:
        for i in range(1, 6):
            alt = result_dir / f'iteration_{i}' / 'code.py'
            if alt.exists() and i != best_iter:
                with open(alt) as f:
                    code = f.read()
                r = sandbox_validate(code)
                if r['ok']:
                    result = r
                    break
        else:
            return None, None
    code_sample = {
        'code': code,
        'revise_state_fn': result['revise_state'],
        'intrinsic_reward_fn': result['intrinsic_reward'],
        'feature_dim': result['feature_dim'],
        'state_dim': result['state_dim'],
    }
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    reward_rules_list = config.get('reward_rules', [])
    valid_rules = []
    for rule in reward_rules_list:
        name = rule if isinstance(rule, str) else rule.get('rule', '')
        params = rule.get('params', {}) if isinstance(rule, dict) else {}
        if name in REWARD_RULE_REGISTRY:
            entry = REWARD_RULE_REGISTRY[name]
            merged = dict(entry['default_params'])
            merged.update(params)
            valid_rules.append({'rule': name, 'params': merged})
    reward_fn = build_reward_rules(valid_rules) if valid_rules else None
    reward_config = {
        'reward_rules': valid_rules,
        'reward_rules_fn': reward_fn,
        'lambda': config.get('lambda', 0.7),
    }
    return code_sample, reward_config


def train(env, agent, max_episodes, seed=None):
    if seed is not None:
        set_seed(seed)
    metrics = {
        'episode_rewards': [], 'episode_sharpes': [],
        'actor_losses': [], 'critic_losses': [],
        'portfolio_values': [], 'wall_times': [],
    }
    all_returns = []
    start_time = time.time()
    for episode in range(max_episodes):
        ep_start = time.time()
        state = env.reset()
        episode_reward = 0
        episode_returns = []
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            weights, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(weights)
            states.append(state); actions.append(weights)
            log_probs.append(log_prob); rewards.append(reward)
            dones.append(float(done))
            episode_reward += reward
            episode_returns.append(info.get('portfolio_return', 0))
            state = next_state
        if len(states) > 1:
            update_info = agent.update(states, actions, log_probs, rewards, dones, state)
            metrics['actor_losses'].append(update_info['actor_loss'])
            metrics['critic_losses'].append(update_info['critic_loss'])
        all_returns.extend(episode_returns)
        ep_time = time.time() - ep_start
        sharpe = sharpe_ratio(all_returns) if len(all_returns) > 20 else 0.0
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_sharpes'].append(sharpe)
        metrics['portfolio_values'].append(float(env.portfolio_value))
        metrics['wall_times'].append(ep_time)
        if (episode + 1) % 10 == 0:
            print(f"    Ep {episode+1}/{max_episodes}: reward={episode_reward:.4f}, sharpe={sharpe:.3f}")
    metrics['total_wall_time'] = time.time() - start_time
    metrics['final_sharpe'] = sharpe_ratio(all_returns)
    metrics['final_mdd'] = max_drawdown(all_returns)
    metrics['final_return'] = (env.portfolio_value - 1.0) * 100
    metrics['final_sortino'] = sortino_ratio(all_returns)
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config_path = PROJECT_DIR / 'configs' / f'config_{args.window}.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    exp_cfg = config.get('experiment', {})
    train_period = tuple(exp_cfg.get('train_period'))
    ppo_cfg = config.get('ppo', {})
    tc = config.get('portfolio', {}).get('transaction_cost', 0.001)
    data_path = config.get('data', {}).get('pickle_file', 'data/portfolio_5stocks.pkl')
    seed = args.seed

    results = {}

    # GIFT-PPO
    code_sample, reward_config = load_best_gift_code(args.window, seed)
    if code_sample is not None:
        print(f"\nGIFT-PPO: state_dim={code_sample['state_dim']}, feature_dim={code_sample['feature_dim']}")
        lam = reward_config.get('lambda', 0.7)
        env_config = dict(config)
        env_config['portfolio'] = dict(config.get('portfolio', {}))
        env_config['portfolio']['default_lambda'] = lam
        env = PortfolioEnv(
            data_path, env_config,
            revise_state_fn=code_sample['revise_state_fn'],
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample['intrinsic_reward_fn'],
            train_period=train_period, transaction_cost=tc,
        )
        agent = PPOAgent(
            state_dim=env.state_dim, seed=seed,
            hidden_dim=ppo_cfg.get('hidden_dim', 256),
            actor_lr=ppo_cfg.get('actor_lr', 3e-4),
            critic_lr=ppo_cfg.get('critic_lr', 3e-4),
            gamma=ppo_cfg.get('gamma', 0.99),
            gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
            clip_epsilon=ppo_cfg.get('clip_epsilon', 0.2),
            entropy_coef=ppo_cfg.get('entropy_coef', 0.01),
            epochs_per_update=ppo_cfg.get('epochs_per_update', 5),
            batch_size=ppo_cfg.get('batch_size', 64),
            use_twin_critic=ppo_cfg.get('use_twin_critic', True),
            value_clip_epsilon=ppo_cfg.get('value_clip_epsilon', 0.2),
            dropout_rate=ppo_cfg.get('dropout_rate', 0.1),
            max_grad_norm=ppo_cfg.get('max_grad_norm', 0.5),
            critic_weight_decay=ppo_cfg.get('critic_weight_decay', 1e-5),
        )
        results['gift'] = train(env, agent, MAX_EPISODES, seed)
        print(f"GIFT-PPO done: Sharpe={results['gift']['final_sharpe']:.3f}, Return={results['gift']['final_return']:.2f}%")
    else:
        print(f"No valid GIFT code for {args.window}")

    # Baseline-PPO
    print(f"\nBaseline-PPO")
    stock_features = [
        {'indicator': 'RSI', 'params': {'window': 14}},
        {'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
        {'indicator': 'Momentum', 'params': {'window': 10}},
        {'indicator': 'Bollinger', 'params': {'window': 20}},
        {'indicator': 'ATR', 'params': {'window': 14}},
    ]
    port_features = [
        {'indicator': 'momentum_rank', 'params': {'window': 20}},
        {'indicator': 'portfolio_volatility', 'params': {'window': 20}},
    ]
    revise_fn = build_revise_state(stock_features)
    port_fn = build_portfolio_features(port_features)
    env = PortfolioEnv(
        data_path, config,
        revise_state_fn=revise_fn, portfolio_features_fn=port_fn,
        detect_regime_fn=detect_market_regime,
        train_period=train_period, transaction_cost=tc,
    )
    print(f"  state_dim={env.state_dim}")
    agent = PPOAgent(
        state_dim=env.state_dim, seed=seed,
        hidden_dim=ppo_cfg.get('hidden_dim', 256),
        actor_lr=ppo_cfg.get('actor_lr', 3e-4),
        critic_lr=ppo_cfg.get('critic_lr', 3e-4),
        gamma=ppo_cfg.get('gamma', 0.99),
        gae_lambda=ppo_cfg.get('gae_lambda', 0.95),
        clip_epsilon=ppo_cfg.get('clip_epsilon', 0.2),
        entropy_coef=ppo_cfg.get('entropy_coef', 0.01),
        epochs_per_update=ppo_cfg.get('epochs_per_update', 5),
        batch_size=ppo_cfg.get('batch_size', 64),
        use_twin_critic=ppo_cfg.get('use_twin_critic', True),
        value_clip_epsilon=ppo_cfg.get('value_clip_epsilon', 0.2),
        dropout_rate=ppo_cfg.get('dropout_rate', 0.1),
        max_grad_norm=ppo_cfg.get('max_grad_norm', 0.5),
        critic_weight_decay=ppo_cfg.get('critic_weight_decay', 1e-5),
    )
    results['baseline'] = train(env, agent, MAX_EPISODES, seed)
    print(f"Baseline done: Sharpe={results['baseline']['final_sharpe']:.3f}, Return={results['baseline']['final_return']:.2f}%")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
