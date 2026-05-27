"""
LLM Prompt Templates for Portfolio Optimization GIFT

Three code-generation prompts:
1. init_prompt: First iteration - state description + building blocks + code example
2. cot_prompt: After training - code + IC analysis + market context
3. next_iteration_prompt: Subsequent iterations - history + suggestions

Plus reward_config_prompt (kept as JSON selection) and _extract_python_code.
"""

import json
import re


def _fmt(val, fmt_str):
    """Safe format: apply format if numeric, else return string as-is."""
    if isinstance(val, (int, float)):
        return format(val, fmt_str)
    return str(val)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown wrapping)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    pattern = r'```(?:json)?\s*\n?(.*?)\n?\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = -1
    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


def _extract_python_code(text: str) -> str:
    """Extract Python code from LLM response.

    Tries:
      1. Extract from ```python ... ``` blocks
      2. Look for 'import numpy' to last function end
      3. Return full text as fallback
    """
    pattern = r'```(?:python)?\s*\n(.*?)\n\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    idx = text.find('import numpy')
    if idx >= 0:
        return text[idx:].strip()

    idx = text.find('def revise_state')
    if idx >= 0:
        import_idx = text.rfind('import', 0, idx)
        if import_idx >= 0:
            line_start = text.rfind('\n', 0, import_idx) + 1
            return text[line_start:].strip()
        return text[idx:].strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Building blocks description for prompts
# ---------------------------------------------------------------------------

BUILDING_BLOCKS_DESC = """
Available computation functions (import from feature_library):

1. compute_relative_momentum(prices, window=20)
   Input: prices = 1D array of close prices (length >= window)
   Output: scalar, this stock's excess return vs window-average
   Use case: identify relatively outperforming stocks

2. compute_realized_volatility(returns, window=20)
   Input: returns = 1D array of daily returns
   Output: scalar, realized volatility
   Use case: measure individual stock risk

3. compute_downside_risk(returns, window=20)
   Input: returns = 1D array of daily returns
   Output: scalar, downside semi-deviation
   Use case: measure downside risk

4. compute_multi_horizon_momentum(prices, windows=[5, 10, 20])
   Input: prices = 1D array of close prices
   Output: array of 3 scalars, momentum at each horizon
   Use case: capture trend at multiple time scales

5. compute_zscore_price(prices, window=20)
   Input: prices = 1D array
   Output: scalar, z-score of current price vs N-day mean
   Use case: mean reversion signal

6. compute_mean_reversion_signal(prices, window=20)
   Input: prices = 1D array
   Output: scalar, mean reversion strength
   Use case: identify overextended prices

7. compute_turnover_ratio(volumes, window=20)
   Input: volumes = 1D array
   Output: scalar, current volume / average volume
   Use case: liquidity detection

Note: compute_cross_sectional_rank and compute_beta are portfolio-level
functions — they require data from all stocks, not available inside revise_state(s).
They are computed at the environment level, not inside your code.
"""


STATE_LAYOUT_DESC = """
The current state for each stock is represented by a 120-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
- `s[0]` through `s[5]`: [close, open, high, low, volume, adjusted_close] for day 1
- `s[6]` through `s[11]`: same 6 channels for day 2
- ...
- `s[114]` through `s[119]`: same 6 channels for day 20

In other words:
- `s[0::6]` = close prices (20 values, oldest to newest)
- `s[1::6]` = open prices
- `s[2::6]` = high prices
- `s[3::6]` = low prices
- `s[4::6]` = volumes
- `s[5::6]` = adjusted close prices
"""


REWARD_RULES = {
    'penalize_concentration': 'Penalty when any stock weight exceeds max_weight. Default max_weight=0.35, penalty=0.1.',
    'reward_diversification': 'Bonus when holding >= min_stocks above 5%. Default min_stocks=3, bonus=0.05.',
    'penalize_turnover': 'Penalty when turnover > threshold. Default threshold=0.1, penalty=0.15.',
    'regime_defensive': 'Bonus for holding cash when risk_level is high. Default crisis_threshold=0.6, cash_bonus=0.1.',
    'momentum_alignment': 'Bonus when weights correlate with momentum rank. Default bonus=0.05.',
    'volatility_scaling': 'Scale down reward in high-vol regime. Default vol_threshold=0.5, scale=0.5.',
    'drawdown_penalty': 'Penalty when drawdown exceeds threshold. Default dd_threshold=0.1, penalty=0.15.',
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_init_prompt(market_stats: str) -> str:
    """Build initial code-generation prompt for first GIFT iteration."""
    return f"""Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ)
      plus a CASH asset (6 assets total). The goal is to maximize risk-adjusted returns.
=========================================================

{STATE_LAYOUT_DESC}

You should design a task-related state representation based on the source 120 dim to better
for reinforcement training, using the detailed information mentioned above to do some calculations,
and feel free to do complex calculations, and then concat them to the source state.

{BUILDING_BLOCKS_DESC}

Market Statistics:
{market_stats}

Besides, we want you to design an intrinsic reward function based on the revise_state python function.

That is to say, we will:
1. use your revise_state python function to get an updated state: updated_s = revise_state(s)
2. use your intrinsic reward function to get an intrinsic reward: r = intrinsic_reward(updated_s)
3. to better design the intrinsic_reward, we recommend you use some source dim in the updated_s,
   which is between updated_s[0] and updated_s[119]
4. however, you must use the extra dim in your given revise_state python function,
   which is between updated_s[120] and the end of updated_s

IMPORTANT for intrinsic_reward design:
- The intrinsic_reward should help the RL agent learn BETTER states, not just amplify returns.
- If the Market Strategy Guidance above indicates high risk (Defensive/Crisis regime),
  your intrinsic_reward should penalize states with high volatility or downside risk.
  Example: reward = -alpha * max(0, extra_risk_feature - threshold)
- If the regime is favorable (Aggressive/Balanced), reward exploration of informative features.
- A good intrinsic_reward correlates with (but is not identical to) future portfolio performance.

Your task is to create two Python functions, named `revise_state`, which takes the current state `s`
as input and returns an updated state representation, and named `intrinsic_reward`, which takes the
updated state `updated_s` as input and returns an intrinsic reward. The functions should be executable
and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative
example of the expected output:

```python
import numpy as np
def revise_state(s):
    # Your state revision implementation goes here
    return updated_s
def intrinsic_reward(updated_s):
    # Your intrinsic reward code implementation goes here
    return intrinsic_reward
```"""


def build_cot_prompt(sample_results_text: str,
                     market_period_summary: str = "") -> str:
    """Build COT feedback prompt after training."""
    return f"""We have successfully trained Reinforcement Learning (RL) policy using different
state revision codes and intrinsic reward function codes sampled by you, and each pair of code
is associated with the training of a policy relatively.

Throughout every state revision code's training process, we monitored:
1. The final policy performance (accumulated reward).
2. Most importantly, every state revise dim's Information Coefficient (IC) with forward returns.
   The IC measures how predictive each state dimension is for future portfolio returns.
   Higher |IC| means the dimension is more useful for the RL agent's decision making.
3. Market environment context and regime-specific IC analysis.

Here are the results:
{sample_results_text}

[Market Environment During Training]
{market_period_summary}

You should analyze the results mentioned above and give suggestions about how to improve the
performance of the "state revision code".

Here are some tips for how to analyze the results:
(a) if you find a state revision code's performance is very low, then you should analyze to
    figure out why it fails
(b) if you find some dims' IC are more related to the final performance, you should analyze
    to figure out what makes it successful
(c) pay attention to regime-specific IC - features that work in trending markets may fail in
    volatile markets
(d) analyze how to improve both the "state revision code" and "intrinsic reward code"
(e) if the Risk Feature Gap Analysis shows features failing in volatile regime, you MUST
    add defensive features (realized_volatility, downside_risk, or zscore_price)
(f) adjust intrinsic_reward to penalize high-risk states when market conditions are unfavorable

Lets think step by step. Your solution should aim to improve the overall performance of the RL policy."""


def build_next_iteration_prompt(market_stats: str,
                                history_text: str,
                                cot_suggestions: str = "") -> str:
    """Build prompt for subsequent GIFT iterations."""
    return f"""Revise the state representation for a reinforcement learning agent.
=========================================================
Task: Dynamic portfolio allocation across 5 stocks (TSLA, NFLX, AMZN, MSFT, JNJ) + CASH.
=========================================================

{STATE_LAYOUT_DESC}

{BUILDING_BLOCKS_DESC}

Updated Market Statistics:
{market_stats}

For this problem, we have some history experience for you, here are some state revision codes
we have tried in the former iterations:
{history_text}

{cot_suggestions}

Based on the former suggestions. We are seeking an improved state revision code and an improved
intrinsic reward code that can enhance the model's performance on the task.

NOTE for intrinsic_reward: Follow the Market Strategy Guidance above. In high-risk regimes,
penalize volatile states; in favorable regimes, reward exploration of informative features.

Your task is to create two Python functions, named `revise_state`, which takes the current state `s`
as input and returns an updated state representation, and named `intrinsic_reward`, which takes the
updated state `updated_s` as input and returns an intrinsic reward. The functions should be executable
and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative
example of the expected output:

```python
import numpy as np
def revise_state(s):
    return updated_s
def intrinsic_reward(updated_s):
    return float_reward
```"""


def build_reward_config_prompt(market_stats: str, iteration: int,
                               history: list = None,
                               feature_rationale: str = "",
                               strategy_hint: str = "") -> str:
    """Build reward rule configuration prompt for LLM (unchanged JSON paradigm)."""
    rules_cat = "\n".join(f"  - {k}: {v}" for k, v in REWARD_RULES.items())

    history_section = ""
    if history:
        history_section = "## Previous Iteration Results\n"
        for h in history[-3:]:
            history_section += f"### Iteration {h.get('iteration', '?')}\n"
            history_section += f"- Reward rules used: {h.get('reward_rules', 'N/A')}\n"
            history_section += f"- Sharpe: {_fmt(h.get('sharpe', 'N/A'), '.3f')}\n"
            history_section += f"- Max Drawdown: {_fmt(h.get('max_drawdown', 'N/A'), '.2f')}%\n"
            turnover = h.get('avg_turnover', 'N/A')
            turnover_str = f"{turnover:.3f}" if isinstance(turnover, (int, float)) else str(turnover)
            history_section += f"- Turnover: {turnover_str}\n"
            history_section += "\n"

    strategy_section = ""
    if strategy_hint:
        strategy_section = f"""## Market Strategy Guidance (for reward rule selection)
{strategy_hint}

IMPORTANT: Select reward rules that align with the market regime above.
- In Crisis/Defensive regime: prioritize regime_defensive, volatility_scaling, drawdown_penalty
- In Balanced regime: use a mix of concentration and turnover rules
- In Aggressive regime: focus on concentration and diversification rules
"""

    return f"""You are configuring the reward function for a PPO-based portfolio optimizer.

The base reward is Mean-Variance: r = portfolio_return - lambda * drawdown^2
Plus an intrinsic reward designed by another LLM code.

Your task: Select and parameterize additional reward rules to guide the agent.

## Market Statistics
{market_stats}

{strategy_section}

{history_section}

## Selected Features Rationale
{feature_rationale}

## Available Reward Rules
{rules_cat}

## Your Task
Select 2-4 reward rules and set their parameters. Consider:
1. Current market conditions and the strategy guidance above
2. Common failure modes (concentration, excessive turnover, ignoring regime)
3. Complementarity between rules (avoid redundant penalties)
4. In high-risk regimes, use MORE conservative lambda (0.6-1.0) and defensive rules

## Output Format (JSON only)
```json
{{
  "reward_rules": [
    {{"rule": "penalize_concentration", "params": {{"max_weight": 0.35}}}},
    {{"rule": "regime_defensive", "params": {{"crisis_threshold": 0.6}}}}
  ],
  "lambda": 0.5,
  "rationale": "Brief explanation of why these rules were selected"
}}
```

Rules:
- Select 2-4 reward rules
- lambda controls risk aversion in base reward (0.1=aggressive, 1.0=conservative)
- In Crisis/Defensive regime, prefer lambda >= 0.6
- Use default params unless market conditions suggest otherwise
"""
