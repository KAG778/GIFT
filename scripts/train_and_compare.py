"""
Train GIFT-PPO vs Baseline-PPO and compare training characteristics.

For each window (W1-W6):
  1. Load best GIFT code (revise_state + intrinsic_reward + reward rules)
  2. Train PPO with GIFT features -> record per-episode metrics
  3. Train PPO baseline (fixed features, no GIFT) -> record per-episode metrics
  4. Generate comparison plots and summary tables

Outputs:
  - training_curves/: per-window reward/sharpe/loss curves
  - convergence_comparison.png: convergence speed comparison
  - training_comparison_summary.json: numeric comparison
  - training_comparison_report.txt: text report
"""

import sys
import os
import json
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

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
WINDOWS = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
SEEDS = [123, 42, 789]
RESULTS_BASE = PROJECT_DIR / 'results'
OUTPUT_DIR = PROJECT_DIR / 'training_comparison'
MAX_EPISODES = 50


def load_best_gift_code(window: str, seed: int = 123):
    """Load best GIFT code and reward config from existing results."""
    result_dir = RESULTS_BASE / f'{window}_seed_{seed}_rerun'
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

    if not code_path.exists() or not config_path.exists():
        return None, None

    with open(code_path) as f:
        code = f.read()
    with open(config_path) as f:
        config = json.load(f)

    # Validate code
    result = sandbox_validate(code)
    if not result['ok']:
        print(f"  Warning: best code for {window} failed validation, trying other iterations")
        for i in range(1, 6):
            alt_code_path = result_dir / f'iteration_{i}' / 'code.py'
            if alt_code_path.exists() and i != best_iter:
                with open(alt_code_path) as f:
                    alt_code = f.read()
                alt_result = sandbox_validate(alt_code)
                if alt_result['ok']:
                    code = alt_code
                    result = alt_result
                    alt_config_path = result_dir / f'iteration_{i}' / 'config.json'
                    if alt_config_path.exists():
                        with open(alt_config_path) as f:
                            config = json.load(f)
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

    # Build reward rules
    reward_rules_list = config.get('reward_rules', [])
    valid_rules = []
    for rule in reward_rules_list:
        if isinstance(rule, str):
            name = rule
            params = {}
        elif isinstance(rule, dict):
            name = rule.get('rule', '')
            params = rule.get('params', {})
        else:
            continue
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


def train_one_run(env, agent, max_episodes, seed=None):
    """Train PPO for max_episodes, return per-episode metrics."""
    if seed is not None:
        set_seed(seed)

    metrics = {
        'episode_rewards': [],
        'episode_sharpes': [],
        'actor_losses': [],
        'critic_losses': [],
        'cumulative_returns': [],
        'portfolio_values': [],
        'wall_times': [],
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
            states.append(state)
            actions.append(weights)
            log_probs.append(log_prob)
            rewards.append(reward)
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
        metrics['cumulative_returns'].extend(episode_returns)
        metrics['portfolio_values'].append(env.portfolio_value)
        metrics['wall_times'].append(ep_time)

        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"    Ep {episode+1}/{max_episodes}: reward={episode_reward:.4f}, "
                  f"sharpe={sharpe:.3f}, time={elapsed:.1f}s")

    total_time = time.time() - start_time
    metrics['total_wall_time'] = total_time
    metrics['final_sharpe'] = sharpe_ratio(all_returns)
    metrics['final_mdd'] = max_drawdown(all_returns)
    metrics['final_return'] = (env.portfolio_value - 1.0) * 100
    metrics['final_sortino'] = sortino_ratio(all_returns)

    return metrics


def run_gift_training(config_path, window, seed):
    """Train PPO with GIFT-generated features."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_cfg = config.get('experiment', {})
    train_period = tuple(exp_cfg.get('train_period'))
    ppo_cfg = config.get('ppo', {})
    transaction_cost = config.get('portfolio', {}).get('transaction_cost', 0.001)
    data_path = config.get('data', {}).get('pickle_file', 'data/portfolio_5stocks.pkl')

    code_sample, reward_config = load_best_gift_code(window, seed)
    if code_sample is None:
        print(f"  [SKIP] No valid GIFT code for {window}_seed_{seed}")
        return None

    lam = reward_config.get('lambda', 0.7)
    env_config = dict(config)
    env_config['portfolio'] = dict(config.get('portfolio', {}))
    env_config['portfolio']['default_lambda'] = lam

    env = PortfolioEnv(
        data_path, env_config,
        revise_state_fn=code_sample['revise_state_fn'],
        portfolio_features_fn=None,
        reward_rules_fn=reward_config.get('reward_rules_fn'),
        detect_regime_fn=detect_market_regime,
        intrinsic_reward_fn=code_sample['intrinsic_reward_fn'],
        train_period=train_period,
        transaction_cost=transaction_cost,
    )

    agent = PPOAgent(
        state_dim=env.state_dim,
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
        seed=seed,
    )

    print(f"  GIFT-PPO: state_dim={env.state_dim}, feature_dim={code_sample.get('feature_dim', 0)}")
    return train_one_run(env, agent, MAX_EPISODES, seed=seed)


def run_baseline_training(config_path, seed):
    """Train pure PPO baseline with fixed features."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_cfg = config.get('experiment', {})
    train_period = tuple(exp_cfg.get('train_period'))
    ppo_cfg = config.get('ppo', {})
    transaction_cost = config.get('portfolio', {}).get('transaction_cost', 0.001)
    data_path = config.get('data', {}).get('pickle_file', 'data/portfolio_5stocks.pkl')

    stock_features = [
        {'indicator': 'RSI', 'params': {'window': 14}},
        {'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
        {'indicator': 'Momentum', 'params': {'window': 10}},
        {'indicator': 'Bollinger', 'params': {'window': 20}},
        {'indicator': 'ATR', 'params': {'window': 14}},
    ]
    portfolio_features = [
        {'indicator': 'momentum_rank', 'params': {'window': 20}},
        {'indicator': 'portfolio_volatility', 'params': {'window': 20}},
    ]

    revise_fn = build_revise_state(stock_features)
    port_fn = build_portfolio_features(portfolio_features)

    env = PortfolioEnv(
        data_path, config,
        revise_state_fn=revise_fn,
        portfolio_features_fn=port_fn,
        detect_regime_fn=detect_market_regime,
        train_period=train_period,
        transaction_cost=transaction_cost,
    )

    agent = PPOAgent(
        state_dim=env.state_dim,
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
        seed=seed,
    )

    print(f"  Baseline-PPO: state_dim={env.state_dim}")
    return train_one_run(env, agent, MAX_EPISODES, seed=seed)


def find_convergence_episode(sharpes, threshold_ratio=0.95):
    """Find the first episode where Sharpe reaches threshold_ratio of the best."""
    if not sharpes:
        return MAX_EPISODES
    best = max(sharpes)
    if best <= 0:
        for i, s in enumerate(sharpes):
            if s >= best * threshold_ratio:
                return i + 1
        return MAX_EPISODES
    target = best * threshold_ratio
    for i, s in enumerate(sharpes):
        if s >= target:
            return i + 1
    return MAX_EPISODES


def compute_reward_stability(rewards):
    """Compute reward stability (lower is more stable)."""
    if len(rewards) < 5:
        return float('inf')
    second_half = rewards[len(rewards)//2:]
    return float(np.std(second_half))


def plot_comparison(all_results, output_dir):
    """Generate all comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.dpi'] = 150

    curves_dir = output_dir / 'training_curves'
    curves_dir.mkdir(exist_ok=True)

    windows = sorted(all_results.keys())

    # ====== Per-window training curves ======
    for window in windows:
        wdata = all_results[window]
        if 'gift' not in wdata or 'baseline' not in wdata:
            continue

        gift = wdata['gift']
        base = wdata['baseline']
        episodes = list(range(1, MAX_EPISODES + 1))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{window} Training Comparison: GIFT-PPO vs Baseline-PPO', fontsize=14, fontweight='bold')

        # Reward curve
        ax = axes[0, 0]
        ax.plot(episodes, gift['episode_rewards'], 'b-o', markersize=3, label='GIFT-PPO', alpha=0.8)
        ax.plot(episodes, base['episode_rewards'], 'r-s', markersize=3, label='Baseline-PPO', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sharpe curve
        ax = axes[0, 1]
        ax.plot(episodes, gift['episode_sharpes'], 'b-o', markersize=3, label='GIFT-PPO', alpha=0.8)
        ax.plot(episodes, base['episode_sharpes'], 'r-s', markersize=3, label='Baseline-PPO', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio (Cumulative)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Actor loss
        ax = axes[1, 0]
        if gift['actor_losses']:
            ax.plot(range(1, len(gift['actor_losses'])+1), gift['actor_losses'], 'b-o', markersize=3, label='GIFT-PPO', alpha=0.8)
        if base['actor_losses']:
            ax.plot(range(1, len(base['actor_losses'])+1), base['actor_losses'], 'r-s', markersize=3, label='Baseline-PPO', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Actor Loss')
        ax.set_title('Actor Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Critic loss
        ax = axes[1, 1]
        if gift['critic_losses']:
            ax.plot(range(1, len(gift['critic_losses'])+1), gift['critic_losses'], 'b-o', markersize=3, label='GIFT-PPO', alpha=0.8)
        if base['critic_losses']:
            ax.plot(range(1, len(base['critic_losses'])+1), base['critic_losses'], 'r-s', markersize=3, label='Baseline-PPO', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Critic Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(curves_dir / f'{window}_training_curves.png', bbox_inches='tight')
        plt.close()

    # ====== Multi-window convergence comparison ======
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Training Characteristics: GIFT-PPO (blue) vs Baseline-PPO (red)', fontsize=14, fontweight='bold')

    # Reward convergence (all windows)
    ax = axes[0]
    for window in windows:
        wdata = all_results[window]
        if 'gift' in wdata:
            smoothed = np.convolve(wdata['gift']['episode_rewards'],
                                   np.ones(5)/5, mode='valid')
            ax.plot(range(3, 3+len(smoothed)), smoothed, 'b-', alpha=0.6, linewidth=1.5)
        if 'baseline' in wdata:
            smoothed = np.convolve(wdata['baseline']['episode_rewards'],
                                   np.ones(5)/5, mode='valid')
            ax.plot(range(3, 3+len(smoothed)), smoothed, 'r--', alpha=0.6, linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Smoothed Reward (5-ep MA)')
    ax.set_title('Reward Convergence (All Windows)')
    ax.grid(True, alpha=0.3)
    # Legend proxy
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0],[0], color='b', linewidth=2),
               Line2D([0],[0], color='r', linestyle='--', linewidth=2)],
              ['GIFT-PPO', 'Baseline-PPO'])

    # Sharpe convergence
    ax = axes[1]
    gift_finals = []
    base_finals = []
    for window in windows:
        wdata = all_results[window]
        if 'gift' in wdata:
            ax.plot(range(1, MAX_EPISODES+1), wdata['gift']['episode_sharpes'],
                    'b-', alpha=0.5, linewidth=1)
            gift_finals.append(wdata['gift']['episode_sharpes'][-1])
        if 'baseline' in wdata:
            ax.plot(range(1, MAX_EPISODES+1), wdata['baseline']['episode_sharpes'],
                    'r--', alpha=0.5, linewidth=1)
            base_finals.append(wdata['baseline']['episode_sharpes'][-1])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Convergence (All Windows)')
    ax.grid(True, alpha=0.3)
    ax.legend([Line2D([0],[0], color='b', linewidth=2),
               Line2D([0],[0], color='r', linestyle='--', linewidth=2)],
              ['GIFT-PPO', 'Baseline-PPO'])

    # Bar chart: convergence speed & final performance
    ax = axes[2]
    x = np.arange(len(windows))
    width = 0.35
    gift_convs = []
    base_convs = []
    for window in windows:
        wdata = all_results[window]
        gift_convs.append(find_convergence_episode(wdata['gift']['episode_sharpes']) if 'gift' in wdata else MAX_EPISODES)
        base_convs.append(find_convergence_episode(wdata['baseline']['episode_sharpes']) if 'baseline' in wdata else MAX_EPISODES)
    ax.bar(x - width/2, gift_convs, width, label='GIFT-PPO', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, base_convs, width, label='Baseline-PPO', color='salmon', alpha=0.8)
    ax.set_xlabel('Window')
    ax.set_ylabel('Episodes to 95% Best Sharpe')
    ax.set_title('Convergence Speed')
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_comparison.png', bbox_inches='tight')
    plt.close()

    # ====== Summary bar chart: final metrics ======
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Final Training Metrics Comparison', fontsize=14, fontweight='bold')

    gift_sharpes, base_sharpes = [], []
    gift_returns, base_returns = [], []
    gift_times, base_times = [], []

    for window in windows:
        wdata = all_results[window]
        if 'gift' in wdata:
            gift_sharpes.append(wdata['gift']['final_sharpe'])
            gift_returns.append(wdata['gift']['final_return'])
            gift_times.append(wdata['gift']['total_wall_time'])
        else:
            gift_sharpes.append(0)
            gift_returns.append(0)
            gift_times.append(0)
        if 'baseline' in wdata:
            base_sharpes.append(wdata['baseline']['final_sharpe'])
            base_returns.append(wdata['baseline']['final_return'])
            base_times.append(wdata['baseline']['total_wall_time'])
        else:
            base_sharpes.append(0)
            base_returns.append(0)
            base_times.append(0)

    x = np.arange(len(windows))
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, gift_sharpes, width, label='GIFT-PPO', color='steelblue')
    ax.bar(x + width/2, base_sharpes, width, label='Baseline-PPO', color='salmon')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Final Training Sharpe')
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x - width/2, gift_returns, width, label='GIFT-PPO', color='steelblue')
    ax.bar(x + width/2, base_returns, width, label='Baseline-PPO', color='salmon')
    ax.set_ylabel('Return (%)')
    ax.set_title('Final Training Return')
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    ax.bar(x - width/2, gift_times, width, label='GIFT-PPO', color='steelblue')
    ax.bar(x + width/2, base_times, width, label='Baseline-PPO', color='salmon')
    ax.set_ylabel('Wall-clock Time (s)')
    ax.set_title('Total Training Time')
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'final_metrics_comparison.png', bbox_inches='tight')
    plt.close()

    # ====== Reward stability comparison ======
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    gift_stabs, base_stabs = [], []
    for window in windows:
        wdata = all_results[window]
        gift_stabs.append(compute_reward_stability(wdata['gift']['episode_rewards']) if 'gift' in wdata else 0)
        base_stabs.append(compute_reward_stability(wdata['baseline']['episode_rewards']) if 'baseline' in wdata else 0)
    ax.bar(x - width/2, gift_stabs, width, label='GIFT-PPO', color='steelblue')
    ax.bar(x + width/2, base_stabs, width, label='Baseline-PPO', color='salmon')
    ax.set_ylabel('Reward Std (2nd Half)')
    ax.set_title('Training Reward Stability (Lower = More Stable)')
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_stability.png', bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}")


def generate_report(all_results, output_dir):
    """Generate text report and JSON summary."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TRAINING CHARACTERISTICS COMPARISON: GIFT-PPO vs Baseline-PPO")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Episodes per run: {MAX_EPISODES}")
    report_lines.append("=" * 80)

    summary = {'windows': {}, 'aggregate': {}}

    gift_sharpes_all, base_sharpes_all = [], []
    gift_returns_all, base_returns_all = [], []
    gift_convs_all, base_convs_all = [], []
    gift_times_all, base_times_all = [], []
    gift_stabs_all, base_stabs_all = [], []

    for window in sorted(all_results.keys()):
        wdata = all_results[window]
        report_lines.append(f"\n{'─'*80}")
        report_lines.append(f"  {window}")
        report_lines.append(f"{'─'*80}")

        wsummary = {}

        for method in ['gift', 'baseline']:
            if method not in wdata:
                report_lines.append(f"  {method.upper()}: SKIPPED (no data)")
                continue
            m = wdata[method]
            conv = find_convergence_episode(m['episode_sharpes'])
            stab = compute_reward_stability(m['episode_rewards'])
            reward_improvement = (np.mean(m['episode_rewards'][-10:]) -
                                  np.mean(m['episode_rewards'][:10]))

            report_lines.append(f"\n  {method.upper()}:")
            report_lines.append(f"    Final Sharpe:    {m['final_sharpe']:.3f}")
            report_lines.append(f"    Final Return:    {m['final_return']:.2f}%")
            report_lines.append(f"    Final MDD:       {m['final_mdd']:.2f}%")
            report_lines.append(f"    Final Sortino:   {m['final_sortino']:.3f}")
            report_lines.append(f"    Convergence Ep:  {conv}/{MAX_EPISODES} "
                               f"({conv/MAX_EPISODES*100:.0f}%)")
            report_lines.append(f"    Reward Stability (std): {stab:.4f}")
            report_lines.append(f"    Reward Improvement:     {reward_improvement:.4f}")
            report_lines.append(f"    Best Ep Reward:  {max(m['episode_rewards']):.4f}")
            report_lines.append(f"    Worst Ep Reward: {min(m['episode_rewards']):.4f}")
            report_lines.append(f"    Total Time:      {m['total_wall_time']:.1f}s")
            report_lines.append(f"    Avg Time/Ep:     {m['total_wall_time']/MAX_EPISODES:.2f}s")

            wsummary[method] = {
                'final_sharpe': m['final_sharpe'],
                'final_return': m['final_return'],
                'final_mdd': m['final_mdd'],
                'convergence_episode': conv,
                'reward_stability': stab,
                'reward_improvement': reward_improvement,
                'total_time': m['total_wall_time'],
                'avg_time_per_ep': m['total_wall_time'] / MAX_EPISODES,
            }

            if method == 'gift':
                gift_sharpes_all.append(m['final_sharpe'])
                gift_returns_all.append(m['final_return'])
                gift_convs_all.append(conv)
                gift_times_all.append(m['total_wall_time'])
                gift_stabs_all.append(stab)
            else:
                base_sharpes_all.append(m['final_sharpe'])
                base_returns_all.append(m['final_return'])
                base_convs_all.append(conv)
                base_times_all.append(m['total_wall_time'])
                base_stabs_all.append(stab)

        # Per-window comparison
        if 'gift' in wsummary and 'baseline' in wsummary:
            gift_w = wsummary['gift']
            base_w = wsummary['baseline']
            report_lines.append(f"\n  COMPARISON:")
            report_lines.append(f"    Sharpe:   GIFT={gift_w['final_sharpe']:.3f} vs "
                               f"Base={base_w['final_sharpe']:.3f} "
                               f"(diff={gift_w['final_sharpe']-base_w['final_sharpe']:+.3f})")
            report_lines.append(f"    Return:   GIFT={gift_w['final_return']:.2f}% vs "
                               f"Base={base_w['final_return']:.2f}% "
                               f"(diff={gift_w['final_return']-base_w['final_return']:+.2f}%)")
            report_lines.append(f"    Converge: GIFT={gift_w['convergence_episode']}ep vs "
                               f"Base={base_w['convergence_episode']}ep "
                               f"(GIFT {'faster' if gift_w['convergence_episode'] <= base_w['convergence_episode'] else 'slower'})")
            report_lines.append(f"    Stability: GIFT={gift_w['reward_stability']:.4f} vs "
                               f"Base={base_w['reward_stability']:.4f} "
                               f"(GIFT {'more stable' if gift_w['reward_stability'] <= base_w['reward_stability'] else 'less stable'})")

        summary['windows'][window] = wsummary

    # Aggregate
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"  AGGREGATE SUMMARY ({len(gift_sharpes_all)} windows)")
    report_lines.append(f"{'='*80}")

    def _stats(vals, label):
        if not vals:
            return f"  {label}: N/A"
        return (f"  {label}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, "
                f"min={np.min(vals):.3f}, max={np.max(vals):.3f}")

    report_lines.append(f"\n  Final Sharpe:")
    report_lines.append(_stats(gift_sharpes_all, "GIFT-PPO "))
    report_lines.append(_stats(base_sharpes_all, "Baseline  "))

    report_lines.append(f"\n  Final Return (%):")
    report_lines.append(_stats(gift_returns_all, "GIFT-PPO "))
    report_lines.append(_stats(base_returns_all, "Baseline  "))

    report_lines.append(f"\n  Convergence Episode:")
    report_lines.append(_stats(gift_convs_all, "GIFT-PPO "))
    report_lines.append(_stats(base_convs_all, "Baseline  "))

    report_lines.append(f"\n  Reward Stability (lower=better):")
    report_lines.append(_stats(gift_stabs_all, "GIFT-PPO "))
    report_lines.append(_stats(base_stabs_all, "Baseline  "))

    report_lines.append(f"\n  Training Time (s):")
    report_lines.append(_stats(gift_times_all, "GIFT-PPO "))
    report_lines.append(_stats(base_times_all, "Baseline  "))

    # Win counts
    gift_wins_sharpe = sum(1 for l, b in zip(gift_sharpes_all, base_sharpes_all) if l > b)
    gift_wins_return = sum(1 for l, b in zip(gift_returns_all, base_returns_all) if l > b)
    gift_wins_conv = sum(1 for l, b in zip(gift_convs_all, base_convs_all) if l <= b)
    gift_wins_stab = sum(1 for l, b in zip(gift_stabs_all, base_stabs_all) if l <= b)
    n = len(gift_sharpes_all)

    report_lines.append(f"\n  GIFT-PPO Wins: Sharpe {gift_wins_sharpe}/{n}, "
                       f"Return {gift_wins_return}/{n}, "
                       f"Convergence {gift_wins_conv}/{n}, "
                       f"Stability {gift_wins_stab}/{n}")

    summary['aggregate'] = {
        'gift_mean_sharpe': float(np.mean(gift_sharpes_all)) if gift_sharpes_all else 0,
        'base_mean_sharpe': float(np.mean(base_sharpes_all)) if base_sharpes_all else 0,
        'gift_mean_return': float(np.mean(gift_returns_all)) if gift_returns_all else 0,
        'base_mean_return': float(np.mean(base_returns_all)) if base_returns_all else 0,
        'gift_mean_convergence': float(np.mean(gift_convs_all)) if gift_convs_all else 0,
        'base_mean_convergence': float(np.mean(base_convs_all)) if base_convs_all else 0,
        'gift_mean_stability': float(np.mean(gift_stabs_all)) if gift_stabs_all else 0,
        'base_mean_stability': float(np.mean(base_stabs_all)) if base_stabs_all else 0,
        'gift_mean_time': float(np.mean(gift_times_all)) if gift_times_all else 0,
        'base_mean_time': float(np.mean(base_times_all)) if base_times_all else 0,
        'gift_wins_sharpe': gift_wins_sharpe,
        'gift_wins_return': gift_wins_return,
        'gift_wins_convergence': gift_wins_conv,
        'gift_wins_stability': gift_wins_stab,
        'total_windows': n,
    }

    with open(output_dir / 'training_comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    report_text = '\n'.join(report_lines)
    with open(output_dir / 'training_comparison_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GIFT-PPO vs Baseline-PPO comparison')
    parser.add_argument('--windows', nargs='+', default=WINDOWS, help='Windows to run')
    parser.add_argument('--seeds', nargs='+', type=int, default=[123], help='Seeds (first seed used to load GIFT code)')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--skip_existing', action='store_true', help='Skip if results exist')
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and (output_dir / 'training_comparison_summary.json').exists():
        print("Results exist, skipping. Remove --skip_existing to rerun.")
        return

    all_results = {}

    for window in args.windows:
        print(f"\n{'#'*80}")
        print(f"# {window}")
        print(f"{'#'*80}")

        config_path = PROJECT_DIR / 'configs' / f'config_{window}.yaml'
        if not config_path.exists():
            print(f"  Config not found: {config_path}")
            continue

        seed = args.seeds[0]

        # GIFT-PPO
        print(f"\n  --- GIFT-PPO ({window}, seed={seed}) ---")
        gift_metrics = run_gift_training(str(config_path), window, seed)

        # Baseline-PPO
        print(f"\n  --- Baseline-PPO ({window}, seed={seed}) ---")
        base_metrics = run_baseline_training(str(config_path), seed)

        window_result = {}
        if gift_metrics is not None:
            window_result['gift'] = gift_metrics
        if base_metrics is not None:
            window_result['baseline'] = base_metrics

        all_results[window] = window_result

    # Save raw metrics
    serializable = {}
    for w, wdata in all_results.items():
        serializable[w] = {}
        for method, m in wdata.items():
            serializable[w][method] = {k: v for k, v in m.items()}
    with open(output_dir / 'raw_metrics.json', 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(all_results, output_dir)

    # Generate report
    print("\nGenerating report...")
    summary = generate_report(all_results, output_dir)

    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
