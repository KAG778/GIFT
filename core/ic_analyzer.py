"""
IC & SHAP analyzer for GIFT COT feedback.

Computes two per-feature signals:
  - IC: Pearson correlation between each feature dim and forward returns,
    i.e. how predictive the feature is.
  - SHAP: contribution of each feature to the trained critic's ``V(s)``,
    i.e. what the RL policy actually relies on.

Their combination is formatted into a COT feedback prompt that drives the
next LLM iteration -- the "IC-based COT Feedback" step in the GIFT paper.
"""

import numpy as np
import torch
from typing import Dict, List, Optional


def compute_ic_profile(revised_states: np.ndarray,
                       forward_returns: np.ndarray) -> Dict[int, float]:
    """Compute Pearson IC between each extra state dim and forward returns.

    Args:
        revised_states: (N, state_dim) array of revised states.
        forward_returns: (N,) array of forward portfolio returns.

    Returns:
        Dict mapping extra-dim index -> IC value.

    Only the dims at index >= 120 (LLM-generated features) are analysed.
    ``IC > 0`` = positively correlated with forward returns (predictive);
    ``IC < 0`` = negative correlation; ``IC ~= 0`` = no predictive power.
    """
    if revised_states.ndim != 2 or forward_returns.ndim != 1:
        return {}
    if revised_states.shape[0] != forward_returns.shape[0]:
        return {}
    if revised_states.shape[0] < 10:
        return {}

    n_dims = revised_states.shape[1]
    extra_start = 120
    if n_dims <= extra_start:
        return {}

    ic_profile = {}
    for dim in range(extra_start, n_dims):
        col = revised_states[:, dim]
        if np.std(col) < 1e-10:
            ic_profile[dim] = 0.0
            continue
        ret_std = np.std(forward_returns)
        if ret_std < 1e-10:
            ic_profile[dim] = 0.0
            continue
        corr = np.corrcoef(col, forward_returns)[0, 1]
        ic_profile[dim] = float(corr) if not np.isnan(corr) else 0.0

    return ic_profile


def compute_regime_specific_ic(revised_states: np.ndarray,
                               forward_returns: np.ndarray,
                               regime_labels: np.ndarray) -> Dict[str, Dict[int, float]]:
    """Compute IC separately per market regime.

    Regimes: ``trending_up`` / ``trending_down`` / ``volatile`` / ``neutral``.
    Used to surface how a feature's predictive power varies with market state.
    """
    result = {}
    for regime in set(regime_labels):
        mask = regime_labels == regime
        if mask.sum() < 10:
            continue
        result[regime] = compute_ic_profile(revised_states[mask], forward_returns[mask])
    return result


def compute_ic_profile_ensemble(revised_states_per_ticker: Dict[str, np.ndarray],
                                forward_returns: np.ndarray) -> Dict[int, float]:
    """Compute IC averaged across all tickers."""
    all_ics = []
    for ticker, states in revised_states_per_ticker.items():
        ic = compute_ic_profile(states, forward_returns)
        if ic:
            all_ics.append(ic)

    if not all_ics:
        return {}

    all_dims = set()
    for ic in all_ics:
        all_dims.update(ic.keys())

    return {dim: float(np.mean([ic.get(dim, 0.0) for ic in all_ics]))
            for dim in sorted(all_dims)}


def compute_regime_specific_ic_ensemble(revised_states_per_ticker: Dict[str, np.ndarray],
                                        forward_returns: np.ndarray,
                                        regime_labels: np.ndarray) -> Dict[str, Dict[int, float]]:
    """Compute regime-specific IC averaged across all tickers."""
    per_ticker_regime_ics = []
    for ticker, states in revised_states_per_ticker.items():
        ric = compute_regime_specific_ic(states, forward_returns, regime_labels)
        if ric:
            per_ticker_regime_ics.append(ric)

    if not per_ticker_regime_ics:
        return {}

    all_regimes = set()
    for ric in per_ticker_regime_ics:
        all_regimes.update(ric.keys())

    result = {}
    for regime in all_regimes:
        regime_ics = [ric.get(regime, {}) for ric in per_ticker_regime_ics]
        all_dims = set()
        for ic in regime_ics:
            all_dims.update(ic.keys())
        result[regime] = {dim: float(np.mean([ic.get(dim, 0.0) for ic in regime_ics]))
                          for dim in sorted(all_dims)}

    return result


def compute_critic_shap(critic, env_states: np.ndarray,
                        extra_start: int = 50,
                        extra_end: int = None,
                        device: str = 'cpu') -> Dict[int, float]:
    """Compute SHAP values for the trained Critic on extra state dims.

    SHAP measures how much each dimension contributes to the Critic's V(s).
    This reveals what the RL policy ACTUALLY uses, beyond statistical IC.

    We focus on 'extra' dims (LLM-generated features) starting at extra_start.
    In the env state layout: compressed_raw(0:50) + extras(50:50+K) + ...

    Args:
        critic: trained CriticNetwork or TwinCriticNetwork
        env_states: (N, state_dim) array of environment states fed to Critic
        extra_start: index where LLM-generated extra dims begin (default 50)
        device: torch device

    Returns:
        Dict mapping dim index -> mean ``|SHAP|`` value (extra dims only).

    Cross-analysis with IC:
      - high SHAP + high ``|IC|`` -> ideal feature (used and predictive)
      - high SHAP + low IC       -> misused feature (used but uninformative,
        likely overfit)
      - low SHAP + high IC       -> underused feature (predictive but ignored)
    """
    import shap

    if env_states.ndim != 2 or env_states.shape[0] < 20:
        return {}

    n_dims = env_states.shape[1]
    effective_end = extra_end if extra_end is not None else n_dims
    if effective_end <= extra_start:
        return {}

    critic.eval()

    def critic_predict(x):
        with torch.no_grad():
            t = torch.FloatTensor(x).to(device)
            if callable(getattr(critic, 'V1', None)):
                vals = critic.V1(t)
            else:
                vals = critic(t)
            return vals.cpu().numpy().reshape(-1, 1)

    bg_size = min(50, env_states.shape[0])
    bg_idx = np.random.choice(env_states.shape[0], bg_size, replace=False)
    background = env_states[bg_idx]

    explain_size = min(100, env_states.shape[0])
    explain_idx = np.random.choice(env_states.shape[0], explain_size, replace=False)
    explain_states = env_states[explain_idx]

    try:
        explainer = shap.KernelExplainer(critic_predict, background)
        shap_values = explainer.shap_values(explain_states, silent=True)
    except Exception:
        return {}

    if isinstance(shap_values, list):
        shap_arr = np.array(shap_values[0])
    else:
        shap_arr = np.array(shap_values)

    if shap_arr.ndim == 3 and shap_arr.shape[2] == 1:
        shap_arr = shap_arr.squeeze(-1)

    if shap_arr.ndim != 2 or shap_arr.shape[1] != n_dims:
        return {}

    # Only report SHAP for extra dims (LLM-generated features)
    end = extra_end if extra_end is not None else n_dims
    shap_profile = {}
    for dim in range(extra_start, end):
        shap_profile[dim] = float(np.mean(np.abs(shap_arr[:, dim])))

    return shap_profile


def _classify_regime(trend: float, vol: float) -> str:
    """Classify a single timestep's regime from trend / vol values.

    Rules:
      - ``volatile``:        ``vol > 0.6``
      - ``trending_up``:     ``trend > 0.3``
      - ``trending_down``:   ``trend < -0.3``
      - ``neutral``:         otherwise.
    """
    if vol > 0.6:
        return 'volatile'
    elif trend > 0.3:
        return 'trending_up'
    elif trend < -0.3:
        return 'trending_down'
    else:
        return 'neutral'


def build_ic_cot_prompt(
    sample_results: List[Dict],
    best_idx: int,
    strong_threshold: float = 0.05,
    weak_threshold: float = 0.02,
    market_period_summary: str = "",
    training_diagnostics: str = "",
) -> str:
    """Build IC + SHAP COT feedback prompt for LLM.

    Args:
        sample_results: list of dicts with keys:
            'code', 'performance', 'ic_profile', 'shap_profile',
            'regime_ic', 'intrinsic_reward_stats'
        best_idx: index of the best-performing sample
        strong_threshold: IC above this = strong feature
        weak_threshold: IC below this = weak feature
        market_period_summary: text describing training period market conditions
        training_diagnostics: text with reward curve, critic loss info

    Returns:
        Formatted COT feedback string.

    Corresponds to the "COT Feedback Generation" step in the paper. The
    feedback contains: (1) per-sample performance + IC/SHAP analysis,
    (2) regime-specific IC, (3) intrinsic-reward diagnostics, and (4)
    improvement suggestions derived from the IC x SHAP cross-analysis.
    """
    lines = []

    for i, sample in enumerate(sample_results):
        code = sample.get('code', '')
        perf = sample.get('performance', {})
        ic_profile = sample.get('ic_profile', {})
        shap_profile = sample.get('shap_profile', {})
        regime_ic = sample.get('regime_ic', {})
        ir_stats = sample.get('intrinsic_reward_stats', {})

        marker = " (BEST)" if i == best_idx else ""
        lines.append(f"========== Code Sample {i+1}{marker} "
                     f"(Sharpe={perf.get('sharpe', 0):.3f}, "
                     f"Return={perf.get('total_return', 0):.2f}%, "
                     f"MaxDD={perf.get('max_drawdown', 0):.2f}%) ==========")
        lines.append(code)
        lines.append("")

        if not ic_profile:
            lines.append("  (No IC profile computed - insufficient data)")
            lines.append("")
            continue

        # Combine IC + SHAP into one table
        all_dims = sorted(set(list(ic_profile.keys()) + list(shap_profile.keys())))
        if shap_profile:
            lines.append("  [Feature Analysis: IC + Critic SHAP]")
            lines.append("    IC = correlation with forward returns (predictive power)")
            lines.append("    SHAP = how much the RL critic relies on this feature")
            lines.append("    (High SHAP + High |IC| = ideal feature)")
            lines.append("")
            for dim in all_dims:
                ic_val = ic_profile.get(dim, 0.0)
                shap_val = shap_profile.get(dim, 0.0)
                abs_ic = abs(ic_val)
                tag = ""
                if abs_ic > strong_threshold:
                    tag = "Strong"
                elif abs_ic < weak_threshold:
                    tag = "Weak"
                if ic_val < -0.03:
                    tag = (tag + " Negative").strip() if tag else "Negative"
                lines.append(f"    s[{dim}]: IC={ic_val:+.4f} SHAP={shap_val:.6f} {tag}")
        else:
            lines.append("  [IC Analysis]")
            for dim, ic_val in sorted(ic_profile.items()):
                abs_ic = abs(ic_val)
                tag = ""
                if abs_ic > strong_threshold:
                    tag = "Strong"
                elif abs_ic < weak_threshold:
                    tag = "Weak"
                if ic_val < -0.03:
                    tag = (tag + " Negative").strip() if tag else "Negative"
                if tag:
                    tag = "<- " + tag
                lines.append(f"    s[{dim}]: IC = {ic_val:+.4f} {tag}")

        if regime_ic:
            lines.append("  [Regime-Specific IC]")
            for regime, dim_ics in regime_ic.items():
                lines.append(f"    {regime}:")
                for dim, ic_val in sorted(dim_ics.items()):
                    lines.append(f"      s[{dim}]: IC = {ic_val:+.4f}")

        if ir_stats:
            lines.append("  [Intrinsic Reward Diagnosis]")
            lines.append(f"    Mean intrinsic_reward = {ir_stats.get('mean', 0):.6f}")
            lines.append(f"    Correlation with performance = "
                         f"{ir_stats.get('correlation_with_performance', 0):.3f}")
            corr = ir_stats.get('correlation_with_performance', 0)
            if abs(corr) > 0.3:
                lines.append(f"    -> Strong guidance effect")
            elif abs(corr) > 0.1:
                lines.append(f"    -> Moderate guidance effect")
            else:
                lines.append(f"    -> Weak guidance effect, consider redesigning")

        lines.append("")

    if training_diagnostics:
        lines.append("[Training Process Diagnostics]")
        lines.append(training_diagnostics)
        lines.append("")

    lines.append("[Improvement Suggestions]")
    best_sample = sample_results[best_idx] if best_idx < len(sample_results) else {}
    best_ic = best_sample.get('ic_profile', {})
    best_shap = best_sample.get('shap_profile', {})
    best_regime_ic = best_sample.get('regime_ic', {})

    if best_ic:
        # Find features that are both IC-strong and SHAP-high (ideal features)
        if best_shap:
            ic_shap_pairs = []
            for dim in best_ic:
                ic_val = best_ic[dim]
                shap_val = best_shap.get(dim, 0.0)
                ic_shap_pairs.append((dim, ic_val, shap_val))

            # Sort by combined score: |IC| * SHAP
            ic_shap_pairs.sort(key=lambda x: abs(x[1]) * x[2], reverse=True)
            if ic_shap_pairs:
                d, ic_v, sh_v = ic_shap_pairs[0]
                lines.append(f"  (a) Most valuable feature: s[{d}] "
                             f"(IC={ic_v:+.4f}, SHAP={sh_v:.6f}). "
                             "The critic relies on it and it predicts returns. "
                             "Consider building derivatives or multi-horizon versions.")

            # Features with high SHAP but low IC — policy uses them but they're useless
            misaligned = [(d, ic, sh) for d, ic, sh in ic_shap_pairs
                          if sh > 0.001 and abs(ic) < weak_threshold]
            if misaligned:
                dim_str = ", ".join(f"s[{d}](SHAP={sh:.4f},IC={ic:+.4f})"
                                    for d, ic, sh in misaligned)
                lines.append(f"  (b) Misused features (high SHAP, low IC): {dim_str}. "
                             "The policy relies on these but they don't predict returns. "
                             "This is a sign of overfitting. Consider removing them.")

            # Features with high IC but low SHAP — useful but policy ignores them
            underused = [(d, ic, sh) for d, ic, sh in ic_shap_pairs
                         if abs(ic) > strong_threshold and sh < 0.0005]
            if underused:
                dim_str = ", ".join(f"s[{d}](IC={ic:+.4f},SHAP={sh:.6f})"
                                    for d, ic, sh in underused)
                lines.append(f"  (c) Underused features (high IC, low SHAP): {dim_str}. "
                             "These predict returns but the policy ignores them. "
                             "Try making them more prominent or combining with existing features.")
        else:
            dims_sorted = sorted(best_ic.items(), key=lambda x: abs(x[1]), reverse=True)
            if dims_sorted:
                best_dim, best_ic_val = dims_sorted[0]
                lines.append(f"  (a) Strongest feature: s[{best_dim}] (IC={best_ic_val:+.4f}). "
                             "Consider building derivatives or multi-horizon versions.")

            weak_dims = [(d, v) for d, v in best_ic.items() if abs(v) < weak_threshold]
            if weak_dims:
                dim_str = ", ".join(f"s[{d}]" for d, _ in weak_dims)
                lines.append(f"  (b) Weak features: {dim_str}. Consider replacing with "
                             "more informative signals (volatility-adjusted momentum, "
                             "cross-sectional rank, etc.).")

        neg_dims = [(d, v) for d, v in best_ic.items() if v < -0.03]
        if neg_dims:
            dim_str = ", ".join(f"s[{d}] (IC={v:+.4f})" for d, v in neg_dims)
            lines.append(f"  (d) Negative IC features: {dim_str}. These may be harmful. "
                         "Consider removing or inverting the signal.")

        if best_regime_ic.get('volatile'):
            volatile_ic = best_regime_ic['volatile']
            n_effective_vol = sum(1 for v in volatile_ic.values() if abs(v) > 0.03)
            if n_effective_vol == 0:
                lines.append("  (e) No features show meaningful IC in volatile regime. "
                             "Consider adding realized_volatility or downside_risk.")

    # --- Risk Feature Gap Analysis (Layer 3: COT feedback) ---
    if best_regime_ic:
        volatile_ic = best_regime_ic.get('volatile', {})
        trending_ic = best_regime_ic.get('trending_up', best_regime_ic.get('trending_down', {}))

        if volatile_ic and best_ic:
            risk_gaps = []
            for dim in best_ic:
                overall_ic = best_ic[dim]
                vol_ic = volatile_ic.get(dim, 0.0)
                # Feature significantly worse in volatile regime
                if abs(overall_ic) > 0.03 and (vol_ic < overall_ic - 0.05):
                    risk_gaps.append((dim, overall_ic, vol_ic))

            if risk_gaps:
                lines.append("[Risk Feature Gap Analysis]")
                lines.append("  Features that FAIL under volatile market conditions:")
                for dim, overall, vol in risk_gaps:
                    lines.append(f"    s[{dim}]: overall IC={overall:+.4f}, volatile-regime IC={vol:+.4f}")
                lines.append("  These features lose predictive power during market stress.")
                lines.append("  Consider adding: realized_volatility, downside_risk, or zscore_price")
                lines.append("  as a defensive complement to protect against regime shifts.")

        # Check if no features work well in volatile regime
        if volatile_ic:
            n_effective_vol = sum(1 for v in volatile_ic.values() if abs(v) > 0.03)
            n_total = len(volatile_ic)
            if n_total > 0 and n_effective_vol == 0:
                lines.append("[Risk Feature Gap Analysis]")
                lines.append("  WARNING: No features show meaningful IC in volatile regime!")
                lines.append("  The current feature set is blind to market stress.")
                lines.append("  STRONGLY recommend adding: downside_risk or realized_volatility")
                lines.append("  and redesigning intrinsic_reward to penalize high-volatility states.")

    lines.append("")
    return "\n".join(lines)
