"""
Market statistics for LLM prompt injection.

Pre-computes per-stock stats, a correlation matrix, and a regime summary from
training data, formatted as text and injected into LLM prompts:
1. Per-stock profile (sector, daily volatility, 20-day return, interpretation).
2. Pairwise correlation matrix across stocks.
3. Diversification guidance based on the correlation matrix.

Design: every statistic is paired with an interpretation string so the LLM can
ground numerical values in plain English. These stats feed both
``build_init_prompt`` and ``build_next_iteration_prompt`` under the
"Market Statistics" section.
"""

import numpy as np

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']
TICKER_PROFILES = {
    'TSLA': {'sector': 'EV/Tech', 'vol_profile': 'Very high (~4% daily)'},
    'NFLX': {'sector': 'Streaming', 'vol_profile': 'Medium-high (~2.5% daily)'},
    'AMZN': {'sector': 'E-commerce/Cloud', 'vol_profile': 'Medium (~2.2% daily)'},
    'MSFT': {'sector': 'Software', 'vol_profile': 'Low-medium (~1.8% daily)'},
    'JNJ': {'sector': 'Pharma', 'vol_profile': 'Low (~1.2% daily, defensive)'},
}


def _extract_closes(s: np.ndarray) -> np.ndarray:
    n = len(s) // 6
    return np.array([s[i * 6] for i in range(n)], dtype=float)


def _extract_volumes(s: np.ndarray) -> np.ndarray:
    n = len(s) // 6
    return np.array([s[i * 6 + 4] for i in range(n)], dtype=float)


def get_market_stats(training_states: dict) -> str:
    """Compute full market statistics with interpretation for LLM prompt.

    Args:
        training_states: dict {ticker: array_of_120d_states (N, 120)} or
                         dict {ticker: 120d_raw_state} for single snapshot.

    Returns:
        Formatted markdown text that can be injected directly into an LLM
        prompt. Contains: (1) per-stock profile table, (2) 5x5 correlation
        matrix, (3) average correlation and diversification guidance.
    """
    lines = []

    # Per-stock stats
    lines.append("### Per-Stock Profile")
    lines.append("| Ticker | Sector | Daily Vol | 20d Return | Interpretation |")
    lines.append("|--------|--------|-----------|------------|----------------|")

    for ticker in TICKERS:
        states = training_states.get(ticker)
        if states is None:
            continue
        if states.ndim == 1:
            states = states.reshape(1, -1)

        closes_list = []
        vols_list = []
        for s in states:
            c = _extract_closes(s)
            closes_list.append(c)
            vols_list.append(_extract_volumes(s))

        # Compute returns within each snapshot to avoid spurious boundary returns
        returns_list = []
        for c in closes_list:
            if len(c) > 1:
                returns_list.append(np.diff(c) / (c[:-1] + 1e-10))
        all_returns_for_vol = np.concatenate(returns_list) if returns_list else np.array([0.0])
        daily_vol = float(np.std(all_returns_for_vol)) * 100 if len(all_returns_for_vol) > 1 else 0.0

        # Use last snapshot for 20d return (most recent 20 consecutive days)
        last_closes = closes_list[-1] if closes_list else np.array([100.0])
        ret_20d = 0.0
        if len(last_closes) >= 21 and last_closes[-21] != 0:
            ret_20d = (last_closes[-1] - last_closes[-21]) / abs(last_closes[-21]) * 100

        avg_vol = float(np.mean(np.concatenate(vols_list))) if vols_list else 1e6

        profile = TICKER_PROFILES.get(ticker, {})
        sector = profile.get('sector', 'Unknown')
        vol_desc = profile.get('vol_profile', 'Unknown')

        # Interpretation
        if daily_vol > 3.0:
            interp = f"High vol ({vol_desc}). Good for trend-following, but risky. Consider concentration limits."
        elif daily_vol > 2.0:
            interp = f"Medium vol ({vol_desc}). Balanced risk/reward. Core holding candidate."
        elif daily_vol > 1.5:
            interp = f"Lower vol ({vol_desc}). Stable performer. Good anchor stock."
        else:
            interp = f"Low vol ({vol_desc}). Defensive. Hedge against downturns. Useful when risk_level is high."

        if ret_20d > 3:
            interp += " Strong recent momentum."
        elif ret_20d < -3:
            interp += " Recent weakness -- consider reducing weight."

        lines.append(f"| {ticker} | {sector} | {daily_vol:.1f}% | {ret_20d:+.1f}% | {interp} |")

    lines.append("")

    # Correlation matrix
    lines.append("### Correlation Matrix (20-day rolling returns)")
    all_returns = {}
    for ticker in TICKERS:
        states = training_states.get(ticker)
        if states is None:
            continue
        if states.ndim == 1:
            states = states.reshape(1, -1)
        # Compute returns within each snapshot to avoid spurious boundary returns
        returns_list = []
        for s in states:
            c = _extract_closes(s)
            if len(c) > 1:
                returns_list.append(np.diff(c) / (c[:-1] + 1e-10))
        if returns_list:
            all_returns[ticker] = np.concatenate(returns_list)

    if len(all_returns) == 5:
        header = "        " + "  ".join(f"{t:>5s}" for t in TICKERS)
        lines.append(header)
        pair_corrs = {}
        for i, t1 in enumerate(TICKERS):
            row = f"{t1:>5s}   "
            for j, t2 in enumerate(TICKERS):
                if i == j:
                    row += "  1.00"
                elif t1 in all_returns and t2 in all_returns:
                    r1, r2 = all_returns[t1], all_returns[t2]
                    n = min(len(r1), len(r2))
                    if n > 5:
                        c = np.corrcoef(r1[-n:], r2[-n:])[0, 1]
                        c = 0.0 if np.isnan(c) else c
                    else:
                        c = 0.0
                    row += f"  {c:.2f}"
                    if i < j:
                        pair_corrs[(t1, t2)] = c
                else:
                    row += "   N/A"
            lines.append(row)

        avg_corr = np.mean(list(pair_corrs.values())) if pair_corrs else 0.0
        lines.append(f"\nAverage pairwise correlation: {avg_corr:.2f}")
        if avg_corr < 0.3:
            lines.append("-> Low correlation: excellent diversification opportunity")
        elif avg_corr < 0.5:
            lines.append("-> Moderate correlation: some diversification benefit exists")
        else:
            lines.append("-> High correlation: limited diversification -- consider sector-exposure rules")

        if pair_corrs:
            min_pair = min(pair_corrs, key=pair_corrs.get)
            max_pair = max(pair_corrs, key=pair_corrs.get)
            lines.append(f"Lowest pair: {min_pair[0]}-{min_pair[1]} ({pair_corrs[min_pair]:.2f}) -> Most diversification value")
            lines.append(f"Highest pair: {max_pair[0]}-{max_pair[1]} ({pair_corrs[max_pair]:.2f}) -> Limited diversification between them")
        lines.append("")

    # --- Strategy hint: dynamic injection based on market conditions ---
    strategy_hint = _compute_strategy_hint(training_states, all_returns)
    if strategy_hint:
        lines.append(strategy_hint)

    return "\n".join(lines)


def compute_strategy_hint(training_states: dict) -> str:
    """Public API: compute strategy hint from training states alone.

    Computes all_returns internally and delegates to _compute_strategy_hint.
    Used by gift_controller to pass strategy context to reward_config_prompt.
    """
    all_returns = {}
    for ticker in TICKERS:
        states = training_states.get(ticker)
        if states is None:
            continue
        if states.ndim == 1:
            states = states.reshape(1, -1)
        closes = np.concatenate([_extract_closes(s) for s in states])
        if len(closes) > 1:
            all_returns[ticker] = np.diff(closes) / (closes[:-1] + 1e-10)
    return _compute_strategy_hint(training_states, all_returns)


def _compute_strategy_hint(training_states: dict, all_returns: dict) -> str:
    """Analyze market conditions and return a strategy hint for LLM prompt injection.

    Uses three signals for regime classification:
    1. regime_detector output (trend_direction, volatility_level, risk_level) — consistent with env
    2. Adaptive thresholds via data quantiles — not hardcoded
    3. Volatility trend (recent vs historical) — detects deteriorating conditions

    Args:
        training_states: dict {ticker: array_of_120d_states}
        all_returns: dict {ticker: returns_array} from correlation computation

    Returns:
        Formatted strategy hint string, or empty string if data insufficient.
    """
    from regime_detector import detect_market_regime

    # --- Signal 1: Regime detector (consistent with environment) ---
    # detect_market_regime expects {ticker: 120d_array}, use last snapshot
    single_states = {}
    for ticker in TICKERS:
        states = training_states.get(ticker)
        if states is None:
            continue
        if states.ndim == 2:
            single_states[ticker] = states[-1]  # use most recent snapshot
        else:
            single_states[ticker] = states
    regime_vec = detect_market_regime(single_states)
    trend_dir = float(regime_vec[0])    # [-1, +1]
    vol_level = float(regime_vec[1])    # [0, 1]
    risk_level = float(regime_vec[2])   # [0, 1]

    # --- Signal 2: Per-stock stats for quantile thresholds ---
    all_vols = []
    all_rets_20d = []
    per_stock_rets = {}
    for ticker in TICKERS:
        states = training_states.get(ticker)
        if states is None:
            continue
        if states.ndim == 1:
            states = states.reshape(1, -1)
        closes = np.concatenate([_extract_closes(s) for s in states])
        if len(closes) < 2:
            continue
        rets = np.diff(closes) / (closes[:-1] + 1e-10)
        all_vols.append(float(np.std(rets)) * 100)
        per_stock_rets[ticker] = rets
        if len(closes) >= 21 and closes[-21] != 0:
            all_rets_20d.append((closes[-1] - closes[-21]) / abs(closes[-21]) * 100)

    if not all_vols:
        return ""

    avg_daily_vol = float(np.mean(all_vols))
    avg_ret_20d = float(np.mean(all_rets_20d)) if all_rets_20d else 0.0

    # Adaptive thresholds: use quantiles of observed volatilities
    vol_arr = np.array(all_vols)
    if len(vol_arr) >= 3:
        vol_p75 = float(np.percentile(vol_arr, 75))
        vol_p90 = float(np.percentile(vol_arr, 90))
    else:
        vol_p75 = 2.5  # fallback
        vol_p90 = 3.5

    # --- Signal 3: Volatility trend (recent vs full-period) ---
    vol_trend = 1.0
    for ticker, rets in per_stock_rets.items():
        if len(rets) >= 20:
            recent_vol = float(np.std(rets[-5:]))
            full_vol = float(np.std(rets)) + 1e-10
            vol_trend = max(vol_trend, recent_vol / full_vol)

    # --- Average correlation ---
    pair_corrs = []
    tickers_with_rets = [t for t in TICKERS if t in all_returns]
    for i, t1 in enumerate(tickers_with_rets):
        for j, t2 in enumerate(tickers_with_rets):
            if i < j:
                r1, r2 = all_returns[t1], all_returns[t2]
                n = min(len(r1), len(r2))
                if n > 5:
                    c = np.corrcoef(r1[-n:], r2[-n:])[0, 1]
                    if not np.isnan(c):
                        pair_corrs.append(c)
    avg_corr = float(np.mean(pair_corrs)) if pair_corrs else 0.0

    # --- Unified regime classification (using regime_detector + adaptive thresholds) ---
    n_declining = sum(1 for r in all_rets_20d if r < -3)

    if risk_level > 0.7 or (vol_level > 0.7 and n_declining >= 3):
        regime = "Crisis"
    elif risk_level > 0.4 or vol_level > 0.6 or n_declining >= 3:
        regime = "Defensive"
    elif avg_daily_vol > vol_p75 or avg_corr > 0.5:
        regime = "Balanced"
    elif trend_dir > 0.3 and vol_level < 0.4:
        regime = "Aggressive"
    else:
        regime = "Balanced"

    # Vol trend override: even if regime looks OK, rising vol is a warning
    vol_deteriorating = vol_trend > 1.5

    # --- Build hint ---
    lines = []
    lines.append("### Market Strategy Guidance")
    lines.append(f"Current market regime: **{regime}**")
    lines.append(f"  Regime detector: trend={trend_dir:+.2f}, volatility_level={vol_level:.2f}, risk_level={risk_level:.2f}")
    lines.append(f"  Average daily volatility: {avg_daily_vol:.1f}% (adaptive thresholds: P75={vol_p75:.1f}%, P90={vol_p90:.1f}%)")
    lines.append(f"  Volatility trend: {vol_trend:.2f}x {'(RISING - deteriorating!)' if vol_deteriorating else '(stable)'}")
    lines.append(f"  Average 20-day return: {avg_ret_20d:+.1f}%")
    lines.append(f"  Average pairwise correlation: {avg_corr:.2f}")
    lines.append(f"  Stocks in decline (20d ret < -3%): {n_declining}/5")

    if vol_deteriorating and regime not in ("Crisis",):
        lines.append("")
        lines.append(f"**WARNING: Volatility is RISING ({vol_trend:.1f}x above average).**")
        lines.append("Market stress is increasing even though current levels look manageable.")
        lines.append("Consider defensive features as a precaution.")

    if regime == "Crisis":
        lines.append("")
        lines.append("**CRITICAL: Market is under severe stress.**")
        lines.append("Feature selection requirements:")
        lines.append("- You MUST include at least one risk/volatility feature (realized_volatility, downside_risk, or zscore_price)")
        lines.append("- Avoid pure momentum features without risk adjustment")
        lines.append("- intrinsic_reward should PENALIZE high-volatility states:")
        lines.append("  Example: reward = -alpha * max(0, volatility_feature - threshold)")
        lines.append("  This helps the agent learn to reduce exposure in crisis")

    elif regime == "Defensive":
        lines.append("")
        lines.append("**CAUTION: Elevated market risk detected.**")
        lines.append("Feature selection guidance:")
        lines.append("- Include at least one defensive feature (downside_risk, realized_volatility, or zscore_price)")
        lines.append("- Pair trend features with risk features for balanced signals")
        lines.append("- intrinsic_reward should factor in risk: reward informative but low-volatility states")
        lines.append("  Example: reward = info_signal / (risk_feature + epsilon)")

    elif regime == "Balanced":
        lines.append("")
        lines.append("**MODERATE: Mixed market conditions.**")
        lines.append("Feature selection guidance:")
        lines.append("- Prefer complementary signals: one trend + one risk + one volume indicator")
        lines.append("- intrinsic_reward should balance exploration with stability")
        lines.append("  Example: reward = trend_signal * (1 - 0.5 * risk_level)")

    else:  # Aggressive
        lines.append("")
        lines.append("**FAVORABLE: Low volatility, good trend opportunity.**")
        lines.append("Feature selection guidance:")
        lines.append("- Momentum and trend features work well in this regime")
        lines.append("- A zscore_price or mean_reversion_signal can still add value as a safety net")
        lines.append("- intrinsic_reward should encourage exploration of trending states")

    if avg_corr > 0.5:
        lines.append("")
        lines.append(f"Note: High average correlation ({avg_corr:.2f}) suggests limited diversification.")
        lines.append("Consider turnover_ratio to detect regime shifts, as all stocks tend to move together.")

    return "\n".join(lines)
