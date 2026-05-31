"""
GIFT controller for portfolio optimization.

The main iteration loop uses a code-generation paradigm (no JSON selection):
  1. LLM emits ``revise_state`` + ``intrinsic_reward`` Python code.
  2. Code sandbox validates the code (AST + test execution + dim detection).
  3. PPO trains with the validated code (multiple samples in parallel).
  4. Per-dim IC and SHAP values are computed on the trained agent.
  5. IC + SHAP COT feedback is fed back to the LLM for the next iteration.
  6. Final test-set evaluation + comparison against a pure-PPO baseline.

This file implements the full GIFT iterative optimization loop from the paper.
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from prompts import (
    build_init_prompt, build_cot_prompt, build_next_iteration_prompt,
    build_reward_config_prompt, _extract_json, _extract_python_code,
)
from code_sandbox import validate as sandbox_validate
from ic_analyzer import (compute_ic_profile, compute_regime_specific_ic,
                         compute_ic_profile_ensemble, compute_regime_specific_ic_ensemble,
                         compute_critic_shap, build_ic_cot_prompt)
from portfolio_features import PORTFOLIO_INDICATOR_REGISTRY, build_portfolio_features
from reward_rules import REWARD_RULE_REGISTRY, build_reward_rules
from regime_detector import detect_market_regime
from market_stats import get_market_stats, compute_strategy_hint
from portfolio_env import PortfolioEnv
from ppo_agent import PPOAgent
from metrics import sharpe_ratio, max_drawdown, sortino_ratio, calmar_ratio


def _fmt(val, fmt):
    """Format a value, handling non-numeric gracefully."""
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


class GIFTController:
    """Orchestrates the GIFT iteration loop for portfolio optimization.

    Manages LLM calls, code validation, PPO training, IC/SHAP analysis, and
    result persistence.
    """

    def __init__(self, config: dict, experiment_dir: str = 'results/default', seed: int = None):
        self.config = config
        self.seed = seed
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # LLM config
        llm_cfg = config.get('llm', {})
        self.client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY', llm_cfg.get('api_key', '')),
            base_url=llm_cfg.get('base_url', 'https://api.openai.com/v1'),
        )
        self.model = llm_cfg.get('model', 'gpt-4o-mini')
        self.temperature = llm_cfg.get('temperature', 0.7)

        # Experiment config
        exp_cfg = config.get('experiment', {})
        self.max_iterations = exp_cfg.get('max_iterations', 5)
        self.sample_count = exp_cfg.get('sample_count', 3)
        self.no_llm = exp_cfg.get('no_llm', False)

        # PPO config
        ppo_cfg = config.get('ppo', {})
        self.ppo_config = ppo_cfg
        self.max_episodes = ppo_cfg.get('max_episodes', 50)

        # Portfolio config
        port_cfg = config.get('portfolio', {})
        self.transaction_cost = port_cfg.get('transaction_cost', 0.001)
        self.default_lambda = port_cfg.get('default_lambda', 0.5)

        # Data paths
        data_cfg = config.get('data', {})
        self.data_path = data_cfg.get('pickle_file', 'data/portfolio_5stocks.pkl')
        self.tickers = list(data_cfg.get('tickers', []))
        self.growth = list(data_cfg.get('growth', []))
        self.defensive = list(data_cfg.get('defensive', []))

        # Train/val/test periods
        train_period = exp_cfg.get('train_period', ['2018-01-01', '2021-12-31'])
        val_period = exp_cfg.get('val_period', ['2022-01-01', '2022-12-31'])
        test_period = exp_cfg.get('test_period', ['2023-01-01', '2023-12-31'])
        self.train_period = tuple(train_period)
        self.val_period = tuple(val_period)
        self.test_period = tuple(test_period)

        # State tracking
        self.iteration_history = []
        self.best_sharpe = -float('inf')
        self.best_config = None

    def _call_llm(self, prompt: str, system_msg: str = None) -> str:
        """Call the LLM with up to 3 retries.

        Defaults to a quantitative-portfolio-manager system prompt.
        """
        if system_msg is None:
            system_msg = "You are a quantitative portfolio manager. Always respond with valid JSON."
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"  LLM call attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
        raise RuntimeError("LLM call failed after 3 attempts")

    def _generate_code(self, iteration: int, n_samples: int = None) -> List[Dict]:
        """Step 1: LLM generates code, sandbox validates. Returns valid samples.

        Iteration 1 uses ``build_init_prompt``; later iterations use
        ``build_next_iteration_prompt`` (history + COT feedback). Generates
        ``n_samples`` candidate codes per iteration, retains only those that
        pass the sandbox. Falls back to a default code if none survives.
        """
        if n_samples is None:
            n_samples = self.sample_count
        print(f"\n[Iteration {iteration}] Step 1: Code Generation ({n_samples} samples)")

        env_tmp = PortfolioEnv(
            self.data_path, self.config,
            train_period=self.train_period,
            transaction_cost=self.transaction_cost,
        )
        training_states, _ = env_tmp.get_training_states(n_samples=200)
        market_stats = get_market_stats(training_states)

        if iteration == 1:
            prompt = build_init_prompt(market_stats)
        else:
            history_text = self._format_history()
            cot_suggestions = self.last_cot_feedback if hasattr(self, 'last_cot_feedback') else ""
            prompt = build_next_iteration_prompt(market_stats, history_text, cot_suggestions)

        valid_samples = []
        for sample_idx in range(n_samples):
            print(f"  Sampling code {sample_idx + 1}/{n_samples}...")
            try:
                response = self._call_llm(
                    prompt,
                    system_msg="You are a quantitative portfolio manager. Write Python code for revise_state and intrinsic_reward functions.")
                code = _extract_python_code(response)
                result = sandbox_validate(code)

                if result['ok']:
                    print(f"    Valid code: feature_dim={result['feature_dim']}, "
                          f"state_dim={result['state_dim']}")
                    valid_samples.append({
                        'code': code,
                        'revise_state_fn': result['revise_state'],
                        'intrinsic_reward_fn': result['intrinsic_reward'],
                        'feature_dim': result['feature_dim'],
                        'state_dim': result['state_dim'],
                    })
                else:
                    print(f"    Invalid code: {result['errors'][:2]}")
            except Exception as e:
                print(f"    Sample {sample_idx + 1} failed: {e}")

        if not valid_samples:
            print("  No valid code samples, using default")
            valid_samples = [self._default_code_config()]

        return valid_samples

    def _configure_rewards(self, iteration: int, feature_rationale: str = "") -> dict:
        """Step 2: LLM configures reward rules via JSON selection.

        The LLM picks 2-4 rules from the 7 predefined reward rules and assigns
        their parameters.
        """
        print(f"\n[Iteration {iteration}] Step 2: Reward Configuration")

        env_tmp = PortfolioEnv(
            self.data_path, self.config,
            train_period=self.train_period,
            transaction_cost=self.transaction_cost,
        )
        training_states, _ = env_tmp.get_training_states(n_samples=100)
        market_stats = get_market_stats(training_states)
        strategy_hint = compute_strategy_hint(training_states)

        prompt = build_reward_config_prompt(
            market_stats=market_stats,
            iteration=iteration,
            history=self.iteration_history,
            feature_rationale=feature_rationale,
            strategy_hint=strategy_hint,
        )

        response = self._call_llm(prompt)

        try:
            parsed = _extract_json(response)
        except ValueError as e:
            print(f"  JSON parse error: {e}")
            return self._default_reward_config()

        rules_list = parsed.get('reward_rules', [])
        lam = parsed.get('lambda', self.default_lambda)
        rationale = parsed.get('rationale', '')

        valid_rules = []
        for rule in rules_list:
            name = rule.get('rule', '')
            if name in REWARD_RULE_REGISTRY:
                entry = REWARD_RULE_REGISTRY[name]
                merged = dict(entry['default_params'])
                merged.update(rule.get('params', {}))
                for pk, pv in merged.items():
                    if pk in entry['param_ranges']:
                        lo, hi = entry['param_ranges'][pk]
                        merged[pk] = type(pv)(np.clip(pv, lo, hi))
                valid_rules.append({'rule': name, 'params': merged})

        reward_fn = build_reward_rules(valid_rules) if valid_rules else None

        print(f"  Selected: {len(valid_rules)} reward rules, lambda={lam:.2f}")

        return {
            'reward_rules': valid_rules,
            'reward_rules_fn': reward_fn,
            'lambda': lam,
            'rationale': rationale,
        }

    def _train_ppo(self, code_sample: dict, reward_config: dict,
                   portfolio_features_fn=None, override_period=None) -> dict:
        """Step 3: Train PPO with the LLM-generated code functions.

        After training, computes:
        - IC profile: predictive power per extra feature dim.
        - SHAP profile: which dims the policy actually relies on.
        - Training diagnostics: reward trend, critic loss.
        """
        period = override_period if override_period else self.train_period
        print(f"\nStep 3: PPO Training (period: {period[0]} ~ {period[1]})")

        # Inject LLM-selected lambda into env config
        lam = reward_config.get('lambda', self.default_lambda)
        env_config = dict(self.config)
        env_config['portfolio'] = dict(self.config.get('portfolio', {}))
        env_config['portfolio']['default_lambda'] = lam

        env = PortfolioEnv(
            self.data_path, env_config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=portfolio_features_fn,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            train_period=period,
            transaction_cost=self.transaction_cost,
        )

        state_dim = env.state_dim
        print(f"  State dim: {state_dim}")

        hidden_dim = self.ppo_config.get('hidden_dim', 256)
        agent = PPOAgent(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            actor_lr=self.ppo_config.get('actor_lr', 3e-4),
            critic_lr=self.ppo_config.get('critic_lr', 3e-4),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_epsilon=self.ppo_config.get('clip_epsilon', 0.2),
            entropy_coef=self.ppo_config.get('entropy_coef', 0.01),
            epochs_per_update=self.ppo_config.get('epochs_per_update', 10),
            batch_size=self.ppo_config.get('batch_size', 64),
            use_twin_critic=self.ppo_config.get('use_twin_critic', True),
            value_clip_epsilon=self.ppo_config.get('value_clip_epsilon', 0.2),
            dropout_rate=self.ppo_config.get('dropout_rate', 0.1),
            max_grad_norm=self.ppo_config.get('max_grad_norm', 0.5),
            critic_weight_decay=self.ppo_config.get('critic_weight_decay', 1e-5),
            seed=self.seed,
        )

        all_rewards = []
        all_returns = []
        all_actor_losses = []
        all_critic_losses = []
        reward_first_half = []
        reward_second_half = []

        for episode in range(self.max_episodes):
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
                all_actor_losses.append(update_info['actor_loss'])
                all_critic_losses.append(update_info['critic_loss'])

            all_rewards.append(episode_reward)
            all_returns.extend(episode_returns)

            mid = self.max_episodes // 2
            if episode < mid:
                reward_first_half.append(episode_reward)
            else:
                reward_second_half.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_rew = np.mean(all_rewards[-10:])
                sharpe = sharpe_ratio(all_returns[-252:]) if len(all_returns) > 10 else 0.0
                print(f"  Episode {episode+1}/{self.max_episodes}: "
                      f"avg_reward={avg_rew:.4f}, sharpe={sharpe:.3f}")

        final_sharpe = sharpe_ratio(all_returns)
        final_sortino = sortino_ratio(all_returns)
        final_mdd = max_drawdown(all_returns)
        final_calmar = calmar_ratio(all_returns)
        total_return = (env.portfolio_value - 1.0) * 100

        print(f"  Training complete: Sharpe={final_sharpe:.3f}, "
              f"MDD={final_mdd:.2f}%, Return={total_return:.2f}%")

        # Compute IC profile and Critic SHAP
        ic_profile = {}
        shap_profile = {}
        regime_ic = {}
        ir_stats = {}
        try:
            revised_result = env.get_revised_states(300)
            revised_per_ticker = revised_result['revised_states_per_ticker']
            forward_returns = revised_result['forward_returns']
            regime_labels = revised_result['regime_labels']

            if revised_per_ticker and len(next(iter(revised_per_ticker.values()))) > 20:
                ic_profile = compute_ic_profile_ensemble(revised_per_ticker, forward_returns)
                regime_ic = compute_regime_specific_ic_ensemble(
                    revised_per_ticker, forward_returns, regime_labels)

                # Critic SHAP: what does the trained policy actually use?
                device = str(agent.device)
                n_env_samples = min(100, len(env.dates) - 22)
                env_state_indices = np.linspace(20, len(env.dates) - 2,
                                                n_env_samples, dtype=int)
                # Use equal weights for SHAP to avoid stale final-episode weights
                saved_weights = env.weights.copy()
                env.weights = np.ones(6) / 6
                env_states = np.array([env._compute_state(idx) for idx in env_state_indices])
                env.weights = saved_weights
                # Extra dims: only LLM-generated features [50 : 50 + feature_dim * 5]
                feature_dim = code_sample.get('feature_dim', 0)
                extra_end = 50 + feature_dim * 5
                raw_shap_profile = compute_critic_shap(
                    agent.critic, env_states, extra_start=50, extra_end=extra_end,
                    device=device)
                # Map SHAP dims (env-state: 50+s*feature_dim+j) to IC dims (revised-state: 120+j)
                # Average SHAP across 5 stocks for each feature j
                shap_profile = {}
                if raw_shap_profile and feature_dim > 0:
                    for j in range(feature_dim):
                        shap_vals = []
                        for s in range(len(self.tickers)):
                            env_dim = 50 + s * feature_dim + j
                            if env_dim in raw_shap_profile:
                                shap_vals.append(raw_shap_profile[env_dim])
                        if shap_vals:
                            shap_profile[120 + j] = float(np.mean(shap_vals))
                if shap_profile:
                    print(f"  SHAP computed for {len(shap_profile)} extra dims (mapped to IC coords)")

                if code_sample.get('intrinsic_reward_fn'):
                    # Average intrinsic reward stats across tickers
                    first_ticker_states = next(iter(revised_per_ticker.values()))
                    if len(first_ticker_states) > 10:
                        # Compute per-ticker intrinsic reward vs forward returns correlation
                        per_ticker_corrs = []
                        n_ir_samples = min(50, len(forward_returns))
                        for ticker, states in revised_per_ticker.items():
                            ir_vals = [float(code_sample['intrinsic_reward_fn'](s))
                                       for s in states[:n_ir_samples]]
                            if len(ir_vals) > 5:
                                c = float(np.corrcoef(ir_vals, forward_returns[:n_ir_samples])[0, 1])
                                if not np.isnan(c):
                                    per_ticker_corrs.append(c)
                        all_ir_values = []
                        for ticker, states in revised_per_ticker.items():
                            all_ir_values.extend(
                                [float(code_sample['intrinsic_reward_fn'](s)) for s in states[:n_ir_samples]])
                        ir_mean = float(np.mean(all_ir_values))
                        ir_corr = float(np.mean(per_ticker_corrs)) if per_ticker_corrs else 0.0
                        ir_stats = {
                            'mean': ir_mean,
                            'correlation_with_performance': ir_corr,
                        }
        except Exception as e:
            print(f"  IC/SHAP computation skipped: {e}")

        # Build training diagnostics for COT feedback
        training_diagnostics = ""
        if all_rewards:
            trend = "improving" if (np.mean(reward_second_half) > np.mean(reward_first_half)) else "declining"
            training_diagnostics += f"Reward trend: {trend} "
            training_diagnostics += f"(first half avg={np.mean(reward_first_half):.4f}, "
            training_diagnostics += f"second half avg={np.mean(reward_second_half):.4f})\n"
        if all_critic_losses:
            training_diagnostics += f"Critic loss: initial={all_critic_losses[0]:.4f}, "
            training_diagnostics += f"final={all_critic_losses[-1]:.4f}\n"
            if all_critic_losses[-1] > all_critic_losses[0]:
                training_diagnostics += "  -> Critic loss INCREASING: possible overfitting\n"
        if all_actor_losses:
            training_diagnostics += f"Actor loss: initial={all_actor_losses[0]:.4f}, "
            training_diagnostics += f"final={all_actor_losses[-1]:.4f}\n"

        return {
            'agent': agent,
            'env': env,
            'sharpe': final_sharpe,
            'sortino': final_sortino,
            'max_drawdown': final_mdd,
            'calmar': final_calmar,
            'total_return': total_return,
            'episode_rewards': all_rewards,
            'episode_returns': all_returns,
            'actor_losses': all_actor_losses,
            'critic_losses': all_critic_losses,
            'all_returns': all_returns,
            'portfolio_value': env.portfolio_value,
            'ic_profile': ic_profile,
            'shap_profile': shap_profile,
            'regime_ic': regime_ic,
            'intrinsic_reward_stats': ir_stats,
            'training_diagnostics': training_diagnostics,
        }

    def _evaluate(self, agent: PPOAgent, code_sample: dict,
                  reward_config: dict, period: tuple = None,
                  label: str = "Validation") -> dict:
        """Evaluate the agent on a given period (val or test).

        Uses a deterministic policy (the mean of the Dirichlet) and records
        performance metrics plus average weights.
        """
        if period is None:
            period = self.test_period

        print(f"\n  [{label} Evaluation] ({period[0]} ~ {period[1]})")

        lam = reward_config.get('lambda', self.default_lambda)
        eval_config = dict(self.config)
        eval_config['portfolio'] = dict(self.config.get('portfolio', {}))
        eval_config['portfolio']['default_lambda'] = lam

        env = PortfolioEnv(
            self.data_path, eval_config,
            revise_state_fn=code_sample.get('revise_state_fn'),
            portfolio_features_fn=None,
            reward_rules_fn=reward_config.get('reward_rules_fn'),
            detect_regime_fn=detect_market_regime,
            intrinsic_reward_fn=code_sample.get('intrinsic_reward_fn'),
            train_period=period,
            transaction_cost=self.transaction_cost,
        )

        state = env.reset()
        done = False
        returns = []
        weights_history = []

        while not done:
            weights, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(weights)
            returns.append(info.get('portfolio_return', 0))
            weights_history.append(info.get('weights', np.ones(6) / 6).copy())
            state = next_state

        ep_sharpe = sharpe_ratio(returns)
        ep_sortino = sortino_ratio(returns)
        ep_mdd = max_drawdown(returns)
        ep_calmar = calmar_ratio(returns)
        ep_return = (env.portfolio_value - 1.0) * 100
        avg_weights = np.mean(weights_history, axis=0)

        print(f"    Sharpe={ep_sharpe:.3f}, Sortino={ep_sortino:.3f}, "
              f"MDD={ep_mdd:.2f}%, Return={ep_return:.2f}%")
        print(f"    Avg weights: " +
              ", ".join(f"{self.tickers[i]}={avg_weights[i]:.3f}" for i in range(len(self.tickers))) +
              f", CASH={avg_weights[len(self.tickers)]:.3f}")

        return {
            f'{label.lower()}_sharpe': ep_sharpe,
            f'{label.lower()}_sortino': ep_sortino,
            f'{label.lower()}_max_drawdown': ep_mdd,
            f'{label.lower()}_calmar': ep_calmar,
            f'{label.lower()}_total_return': ep_return,
            f'{label.lower()}_returns': returns,
            f'{label.lower()}_avg_weights': {**{self.tickers[i]: float(avg_weights[i]) for i in range(len(self.tickers))},
                                              'CASH': float(avg_weights[len(self.tickers)])},
        }

    def _default_code_config(self) -> dict:
        """Fallback code config used when LLM generation fails.

        Uses the ``compute_relative_momentum`` + ``compute_realized_volatility``
        building blocks.
        """
        code = """import numpy as np
from feature_library import compute_relative_momentum, compute_realized_volatility
def revise_state(s):
    closes = s[0::6]
    returns = np.diff(closes) / (closes[:-1] + 1e-10)
    mom = compute_relative_momentum(closes, 20)
    vol = compute_realized_volatility(returns, 20)
    return np.concatenate([s, [mom, vol]])
def intrinsic_reward(updated_s):
    return 0.01 * abs(updated_s[120]) / (updated_s[121] + 0.01)
"""
        result = sandbox_validate(code)
        if result['ok']:
            return {
                'code': code,
                'revise_state_fn': result['revise_state'],
                'intrinsic_reward_fn': result['intrinsic_reward'],
                'feature_dim': result['feature_dim'],
                'state_dim': result['state_dim'],
            }
        return {
            'code': 'import numpy as np\ndef revise_state(s): return s\ndef intrinsic_reward(s): return 0.0',
            'revise_state_fn': lambda s: s,
            'intrinsic_reward_fn': lambda s: 0.0,
            'feature_dim': 0,
            'state_dim': 120,
        }

    def _default_reward_config(self) -> dict:
        """Fallback reward config."""
        rules = [
            {'rule': 'penalize_concentration', 'params': {'max_weight': 0.35, 'penalty': 0.1}},
            {'rule': 'penalize_turnover', 'params': {'threshold': 0.1, 'penalty': 0.15}},
        ]
        return {
            'reward_rules': rules,
            'reward_rules_fn': build_reward_rules(rules),
            'lambda': self.default_lambda,
            'rationale': 'Default fallback rules',
        }

    def _format_history(self) -> str:
        """Format iteration history for the next-iteration prompt.

        Only the most recent 3 iterations are shown to keep the prompt short.
        """
        lines = []
        for h in self.iteration_history[-3:]:
            lines.append(f"Iteration {h.get('iteration', '?')}:")
            lines.append(f"  Train Sharpe: {_fmt(h.get('sharpe', 'N/A'), '.3f')}")
            lines.append(f"  Return: {_fmt(h.get('total_return', 'N/A'), '.2f')}%")
            if h.get('ic_profile'):
                lines.append(f"  IC profile: {h['ic_profile']}")
            lines.append("")
        return "\n".join(lines)

    def _get_market_summary(self) -> str:
        """Build a brief market summary that is injected into COT feedback.

        Contains annualised return, annualised volatility, and a trend tag.
        """
        try:
            env_tmp = PortfolioEnv(
                self.data_path, self.config,
                train_period=self.train_period,
                transaction_cost=self.transaction_cost,
            )
            training_states, forward_returns = env_tmp.get_training_states(n_samples=100)
            if len(forward_returns) > 10:
                avg_ret = float(np.mean(forward_returns)) * 252 * 100
                vol = float(np.std(forward_returns)) * np.sqrt(252) * 100
                trend = "bullish" if avg_ret > 5 else ("bearish" if avg_ret < -5 else "neutral")
                return (f"Training period: {self.train_period[0]} ~ {self.train_period[1]}\n"
                        f"  Annualized return: {avg_ret:.1f}%\n"
                        f"  Annualized volatility: {vol:.1f}%\n"
                        f"  Trend: {trend}")
        except Exception:
            pass
        return f"Training period: {self.train_period[0]} ~ {self.train_period[1]}"

    def run(self):
        """Run the full GIFT iteration loop with multi-sample code generation.

        Each iteration: code generation -> reward configuration -> multi-sample
        training -> COT feedback -> persist results. After all iterations, the
        best configuration is evaluated on the test set and compared against
        the pure-PPO baseline.
        """
        print("=" * 60)
        print("GIFT Portfolio Optimization (Code-Generation Mode)")
        print(f"Iterations: {self.max_iterations}, Samples: {self.sample_count}, "
              f"PPO episodes: {self.max_episodes}")
        print(f"Train: {self.train_period}, Val: {self.val_period}, Test: {self.test_period}")
        if self.no_llm:
            print("Mode: NO_LLM (using default features & reward rules)")
        print("=" * 60)

        self.last_cot_feedback = ""

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*60}")

            try:
                # First iteration uses more samples for diverse initialization
                n_samples = 3 if iteration == 1 else self.sample_count
                if self.no_llm:
                    code_samples = [self._default_code_config()]
                    reward_config = self._default_reward_config()
                else:
                    code_samples = self._generate_code(iteration, n_samples=n_samples)
                    reward_config = self._configure_rewards(
                        iteration, "Code-generated features")

                sample_results = []
                for s_idx, code_sample in enumerate(code_samples):
                    print(f"\n  --- Training Sample {s_idx + 1}/{len(code_samples)} ---")
                    train_result = self._train_ppo(code_sample, reward_config)

                    sample_results.append({
                        'code': code_sample.get('code', ''),
                        'train_result': train_result,
                        'performance': {
                            'sharpe': train_result['sharpe'],
                            'total_return': train_result['total_return'],
                            'max_drawdown': train_result['max_drawdown'],
                        },
                        'ic_profile': train_result.get('ic_profile', {}),
                        'shap_profile': train_result.get('shap_profile', {}),
                        'regime_ic': train_result.get('regime_ic', {}),
                        'intrinsic_reward_stats': train_result.get('intrinsic_reward_stats', {}),
                    })

                best_idx = max(range(len(sample_results)),
                              key=lambda i: sample_results[i]['performance']['sharpe'])
                best = sample_results[best_idx]
                best_perf = best['performance']
                print(f'  Best sample: {best_idx + 1} '
                      f'(Train Sharpe={best_perf["sharpe"]:.3f}, '
                      f'Return={best_perf["total_return"]:.2f}%)')

                if not self.no_llm and len(sample_results) > 0:
                    best_train_result = sample_results[best_idx].get("train_result", {})
                    cot_text = build_ic_cot_prompt(
                        sample_results, best_idx,
                        market_period_summary=self._get_market_summary(),
                        training_diagnostics=best_train_result.get("training_diagnostics", ""),
                    )
                    self.last_cot_feedback = cot_text
                    print(f"  COT feedback generated ({len(cot_text)} chars)")

                record = {
                    "iteration": iteration,
                    "n_samples": len(code_samples),
                    "best_sample_idx": best_idx,
                    "reward_rules": [r["rule"] for r in reward_config.get("reward_rules", [])],
                    "lambda": reward_config.get("lambda", self.default_lambda),
                    "sharpe": best["performance"]["sharpe"],
                    "max_drawdown": best["performance"]["max_drawdown"],
                    "total_return": best["performance"]["total_return"],
                    "ic_profile": {str(k): f"{v:.4f}" for k, v in best.get("ic_profile", {}).items()},
                }
                self.iteration_history.append(record)

                self._save_iteration(iteration, code_samples[best_idx], reward_config,
                                     best["train_result"], record)

                if best["performance"]["sharpe"] > self.best_sharpe:
                    self.best_sharpe = best["performance"]["sharpe"]
                    self.best_config = {
                        "iteration": iteration,
                        "code": code_samples[best_idx].get("code", ""),
                        "reward_config": reward_config,
                    }
                    model_path = self.experiment_dir / "best_model.pt"
                    best["train_result"]["agent"].save(str(model_path))
                    print(f"  New best model! Train Sharpe={self.best_sharpe:.3f}")


            except Exception as e:
                print(f"  Iteration {iteration} failed: {e}")
                import traceback
                traceback.print_exc()

        self._save_summary()

        # V1 Step 5: Code Transfer + Retrain on first half of test_period
        test_result = {}
        baseline_result = {}
        if self.best_config:
            print(f"\n{'='*60}")
            print("STEP 5: Code Transfer + Retrain on Test Period (V1)")
            print(f"{'='*60}")

            # Recreate code_sample from best config (transfer learned features)
            best_code = self.best_config.get('code', '')
            code_sample = self._default_code_config()
            if best_code:
                try:
                    r = sandbox_validate(best_code)
                    if r['ok']:
                        code_sample = {
                            'code': best_code,
                            'revise_state_fn': r['revise_state'],
                            'intrinsic_reward_fn': r['intrinsic_reward'],
                            'feature_dim': r['feature_dim'],
                            'state_dim': r['state_dim'],
                        }
                except Exception:
                    pass

            reward_config = self.best_config.get('reward_config', self._default_reward_config())

            # Split test_period: first 50% for retraining, last 50% for evaluation
            import pickle
            with open(self.data_path, 'rb') as f:
                raw_data = pickle.load(f)
            all_dates = sorted(raw_data.keys())
            test_dates = [d for d in all_dates
                          if self.test_period[0] <= d <= self.test_period[1]]
            split_idx = len(test_dates) // 2
            retrain_period = (test_dates[0], test_dates[split_idx])
            eval_period = (test_dates[split_idx + 1], test_dates[-1])
            print(f"  Retrain period: {retrain_period[0]} ~ {retrain_period[1]} ({split_idx + 1} days)")
            print(f"  Eval period:    {eval_period[0]} ~ {eval_period[1]} ({len(test_dates) - split_idx - 1} days)")

            # V1: Train a NEW PPO from scratch on retrain_period using transferred code
            print(f"\n  --- Retraining PPO with transferred code on retrain period ---")
            print(f"  Strategy: keep revise_state + intrinsic_reward code, train fresh PPO")
            retrain_result = self._train_ppo(code_sample, reward_config,
                                              override_period=retrain_period)

            # Evaluate the retrained agent on the eval_period
            test_result = self._evaluate(
                retrain_result['agent'], code_sample, reward_config,
                period=eval_period, label="Test")

            # Baseline 1: pure PPO retrain on test first 50%, eval last 50%
            baseline1_result = self._run_baseline_comparison_v1(retrain_period, eval_period)

            # Baseline 2: pure PPO train on train_period + continue on test first 50%, eval last 50%
            baseline2_result = self._run_baseline_train_then_test(retrain_period, eval_period)

            # Print comparison table (3-way)
            self._print_comparison_v1(test_result, baseline1_result, baseline2_result)

            # Save test + baseline results
            summary_extra = {
                'transfer_strategy': 'V1_code_retrain',
                'retrain_period': list(retrain_period),
                'eval_period': list(eval_period),
                'test_result': {k: v for k, v in test_result.items() if 'returns' not in k},
                'baseline1_test_only': {k: v for k, v in baseline1_result.items() if 'returns' not in k},
                'baseline2_train_plus_test': {k: v for k, v in baseline2_result.items() if 'returns' not in k},
            }
            with open(self.experiment_dir / 'final_comparison.json', 'w') as f:
                json.dump(summary_extra, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"GIFT V1 Complete. Best Train Sharpe: {self.best_sharpe:.3f}")

    def _save_iteration(self, iteration, code_sample, reward_config,
                        train_result, record):
        """Save iteration results to disk."""
        iter_dir = self.experiment_dir / f'iteration_{iteration}'
        iter_dir.mkdir(exist_ok=True)

        with open(iter_dir / 'code.py', 'w') as f:
            f.write(code_sample.get('code', ''))

        save_cfg = {
            'reward_rules': reward_config.get('reward_rules', []),
            'lambda': reward_config.get('lambda', self.default_lambda),
            'feature_dim': code_sample.get('feature_dim', 0),
            'state_dim': code_sample.get('state_dim', 120),
        }
        with open(iter_dir / 'config.json', 'w') as f:
            json.dump(save_cfg, f, indent=2, default=str)

        with open(iter_dir / 'metrics.json', 'w') as f:
            json.dump(record, f, indent=2, default=str)

        train_result['agent'].save(str(iter_dir / 'model.pt'))

    def _run_baseline_comparison_v1(self, retrain_period, eval_period) -> dict:
        """V1 Baseline: pure PPO (no LLM) retrain on test first 50%, eval last 50%.

        Pure PPO is trained from scratch on the first 50% of ``test_period``
        and evaluated on the last 50%. This is the fairness baseline: both
        GIFT and baseline train and evaluate on the same data.
        """
        print(f"\n  --- Pure PPO Baseline V1 (retrain on {retrain_period[0]}~{retrain_period[1]}) ---")

        env = PortfolioEnv(
            self.data_path, self.config,
            detect_regime_fn=detect_market_regime,
            train_period=retrain_period,
            transaction_cost=self.transaction_cost,
        )

        state_dim = env.state_dim
        print(f"    Baseline state dim: {state_dim}")

        agent = PPOAgent(
            state_dim=state_dim,
            hidden_dim=self.ppo_config.get('hidden_dim', 256),
            actor_lr=self.ppo_config.get('actor_lr', 3e-4),
            critic_lr=self.ppo_config.get('critic_lr', 3e-4),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_epsilon=self.ppo_config.get('clip_epsilon', 0.2),
            entropy_coef=self.ppo_config.get('entropy_coef', 0.01),
            epochs_per_update=self.ppo_config.get('epochs_per_update', 10),
            batch_size=self.ppo_config.get('batch_size', 64),
            use_twin_critic=self.ppo_config.get('use_twin_critic', True),
            value_clip_epsilon=self.ppo_config.get('value_clip_epsilon', 0.2),
            dropout_rate=self.ppo_config.get('dropout_rate', 0.1),
            max_grad_norm=self.ppo_config.get('max_grad_norm', 0.5),
            critic_weight_decay=self.ppo_config.get('critic_weight_decay', 1e-5),
            seed=self.seed,
        )

        for episode in range(self.max_episodes):
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

        code_sample_none = {'revise_state_fn': None, 'intrinsic_reward_fn': None}
        reward_cfg_none = {'reward_rules_fn': None}
        result = self._evaluate(
            agent, code_sample_none, reward_cfg_none,
            period=eval_period, label="Baseline")
        return result

    def _run_baseline_train_then_test(self, retrain_period, eval_period) -> dict:
        """V1 Baseline 2: pure PPO trained on train + continued on test first 50%, eval last 50%.

        Pure PPO is first trained on ``train_period``, then fine-tuned on the
        first 50% of ``test_period`` and evaluated on the last 50%. Tests
        GIFT's code-transfer advantage over the traditional "train then adapt"
        approach.
        """
        print(f"\n  --- Pure PPO Baseline 2 (train + continue on test first 50%) ---")

        # Phase 1: Train on train_period
        env_train = PortfolioEnv(
            self.data_path, self.config,
            detect_regime_fn=detect_market_regime,
            train_period=self.train_period,
            transaction_cost=self.transaction_cost,
        )
        state_dim = env_train.state_dim
        print(f"    Baseline2 state dim: {state_dim}")
        print(f"    Phase 1: Training on {self.train_period[0]} ~ {self.train_period[1]}")

        agent = PPOAgent(
            state_dim=state_dim,
            hidden_dim=self.ppo_config.get('hidden_dim', 256),
            actor_lr=self.ppo_config.get('actor_lr', 3e-4),
            critic_lr=self.ppo_config.get('critic_lr', 3e-4),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_epsilon=self.ppo_config.get('clip_epsilon', 0.2),
            entropy_coef=self.ppo_config.get('entropy_coef', 0.01),
            epochs_per_update=self.ppo_config.get('epochs_per_update', 10),
            batch_size=self.ppo_config.get('batch_size', 64),
            use_twin_critic=self.ppo_config.get('use_twin_critic', True),
            value_clip_epsilon=self.ppo_config.get('value_clip_epsilon', 0.2),
            dropout_rate=self.ppo_config.get('dropout_rate', 0.1),
            max_grad_norm=self.ppo_config.get('max_grad_norm', 0.5),
            critic_weight_decay=self.ppo_config.get('critic_weight_decay', 1e-5),
            seed=self.seed,
        )

        for episode in range(self.max_episodes):
            state = env_train.reset()
            states, actions, log_probs, rewards, dones = [], [], [], [], []
            done = False
            while not done:
                weights, log_prob = agent.select_action(state)
                next_state, reward, done, info = env_train.step(weights)
                states.append(state)
                actions.append(weights)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(done))
                state = next_state
            if len(states) > 1:
                agent.update(states, actions, log_probs, rewards, dones, state)

        # Phase 2: Continue training on retrain_period (test first 50%)
        print(f"    Phase 2: Continue training on {retrain_period[0]} ~ {retrain_period[1]}")
        env_ft = PortfolioEnv(
            self.data_path, self.config,
            detect_regime_fn=detect_market_regime,
            train_period=retrain_period,
            transaction_cost=self.transaction_cost,
        )
        for episode in range(self.max_episodes):
            state = env_ft.reset()
            states, actions, log_probs, rewards, dones = [], [], [], [], []
            done = False
            while not done:
                weights, log_prob = agent.select_action(state)
                next_state, reward, done, info = env_ft.step(weights)
                states.append(state)
                actions.append(weights)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(done))
                state = next_state
            if len(states) > 1:
                agent.update(states, actions, log_probs, rewards, dones, state)

        # Evaluate on eval_period
        code_sample_none = {'revise_state_fn': None, 'intrinsic_reward_fn': None}
        reward_cfg_none = {'reward_rules_fn': None}
        result = self._evaluate(
            agent, code_sample_none, reward_cfg_none,
            period=eval_period, label="Base2")
        return result

    def _print_comparison_v1(self, test_result: dict, baseline1_result: dict,
                              baseline2_result: dict):
        """Print the V1 three-way comparison table.

        Columns:
        - GIFT+PPO    (code transfer, retrain on test first 50%)
        - Baseline 1  (pure PPO, retrain on test first 50% only)
        - Baseline 2  (pure PPO, train on train + test first 50%)
        """
        def _get(result, suffix, default=float('nan')):
            for prefix in ['test', 'base2', 'baseline', 'val']:
                v = result.get(f'{prefix}_{suffix}')
                if v is not None:
                    return v
            return default

        print(f"\n{'='*80}")
        print(f"FINAL COMPARISON V1 (Eval: {self.test_period[0]} ~ {self.test_period[1]})")
        print(f"{'='*80}")
        print(f"{'Metric':<20} {'GIFT+PPO':>12} {'PPO(test)':>12} {'PPO(train+test)':>14}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*14}")

        metrics = [
            ('Total Return', 'total_return', '%.2f%%'),
            ('Sharpe Ratio', 'sharpe', '%.3f'),
            ('Sortino Ratio', 'sortino', '%.3f'),
            ('Max Drawdown', 'max_drawdown', '%.2f%%'),
            ('Calmar Ratio', 'calmar', '%.3f'),
        ]

        for label, suffix, fmt in metrics:
            gift_val = _get(test_result, suffix)
            base1_val = _get(baseline1_result, suffix)
            base2_val = _get(baseline2_result, suffix)
            gift_str = fmt % gift_val if not np.isnan(gift_val) else 'N/A'
            base1_str = fmt % base1_val if not np.isnan(base1_val) else 'N/A'
            base2_str = fmt % base2_val if not np.isnan(base2_val) else 'N/A'
            higher_is_better = (suffix != 'max_drawdown')
            best = max(gift_val, base1_val, base2_val) if higher_is_better else min(gift_val, base1_val, base2_val) if not any(np.isnan(v) for v in [gift_val, base1_val, base2_val]) else float('nan')
            m1 = " *" if not np.isnan(gift_val) and gift_val == best else ""
            m2 = " *" if not np.isnan(base1_val) and base1_val == best else ""
            m3 = " *" if not np.isnan(base2_val) and base2_val == best else ""
            print(f"{label:<20} {gift_str:>12}{m1} {base1_str:>12}{m2} {base2_str:>14}{m3}")

        # Avg weights comparison
        gift_w = _get(test_result, 'avg_weights', {})
        base1_w = _get(baseline1_result, 'avg_weights', {})
        base2_w = _get(baseline2_result, 'avg_weights', {})
        print(f"\n{'Avg Weights':<20} {'GIFT+PPO':>12} {'PPO(test)':>12} {'PPO(train+test)':>14}")
        print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*14}")
        for name in self.tickers + ['CASH']:
            lw = f"{gift_w.get(name, 0):.3f}" if gift_w else 'N/A'
            b1w = f"{base1_w.get(name, 0):.3f}" if base1_w else 'N/A'
            b2w = f"{base2_w.get(name, 0):.3f}" if base2_w else 'N/A'
            print(f"  {name:<18} {lw:>12} {b1w:>12} {b2w:>14}")
        print(f"{'='*80}")
        print("  (* = best on this metric)")

    def _print_comparison(self, test_result: dict, baseline_result: dict):
        """Print side-by-side GIFT+PPO vs pure-PPO comparison table.

        Reports total return, Sharpe, Sortino, max drawdown, Calmar, and
        average weights.
        """
        # Normalize keys: both results use {label}_sharpe format
        # test_result keys: test_sharpe, test_sortino, ...
        # baseline_result keys: baseline_sharpe, baseline_sortino, ...
        def _get(result, suffix, default=float('nan')):
            for prefix in ['test', 'baseline', 'val']:
                v = result.get(f'{prefix}_{suffix}')
                if v is not None:
                    return v
            return default

        print(f"\n{'='*60}")
        print(f"FINAL COMPARISON (Test: {self.test_period[0]} ~ {self.test_period[1]})")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'GIFT+PPO':>12} {'Pure PPO':>12}")
        print(f"{'-'*20} {'-'*12} {'-'*12}")

        metrics = [
            ('Total Return', 'total_return', '%.2f%%'),
            ('Sharpe Ratio', 'sharpe', '%.3f'),
            ('Sortino Ratio', 'sortino', '%.3f'),
            ('Max Drawdown', 'max_drawdown', '%.2f%%'),
            ('Calmar Ratio', 'calmar', '%.3f'),
        ]

        for label, suffix, fmt in metrics:
            gift_val = _get(test_result, suffix)
            base_val = _get(baseline_result, suffix)
            gift_str = fmt % gift_val if not np.isnan(gift_val) else 'N/A'
            base_str = fmt % base_val if not np.isnan(base_val) else 'N/A'
            marker = " *" if not np.isnan(gift_val) and not np.isnan(base_val) and gift_val > base_val else ""
            print(f"{label:<20} {gift_str:>12} {base_str:>12}{marker}")

        # Avg weights comparison
        gift_w = _get(test_result, 'avg_weights', {})
        base_w = _get(baseline_result, 'avg_weights', {})
        print(f"\n{'Avg Weights':<20} {'GIFT+PPO':>12} {'Pure PPO':>12}")
        print(f"{'-'*20} {'-'*12} {'-'*12}")
        for name in self.tickers + ['CASH']:
            lw = f"{gift_w.get(name, 0):.3f}" if gift_w else 'N/A'
            bw = f"{base_w.get(name, 0):.3f}" if base_w else 'N/A'
            print(f"  {name:<18} {lw:>12} {bw:>12}")
        print(f"{'='*60}")
        print("  (* = GIFT+PPO wins on this metric)")

    def _save_summary(self):
        """Save overall experiment summary."""
        summary = {
            'best_train_sharpe': self.best_sharpe,
            'best_iteration': self.best_config['iteration'] if self.best_config else None,
            'iterations': len(self.iteration_history),
            'history': self.iteration_history,
        }
        with open(self.experiment_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
