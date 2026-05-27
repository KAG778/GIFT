# GIFT: LLM-Guided Iterative Feature Tuning for Portfolio PPO

A closed-loop framework that couples LLM-driven feature engineering with PPO-based
portfolio optimization. The LLM iteratively rewrites `revise_state` and
`intrinsic_reward` Python code; a PPO agent trains on the augmented state; per-feature
IC and SHAP diagnostics are fed back to the LLM for the next iteration.

> **Anonymous double-blind submission.** This release does not include author
> identity, affiliation, contact information, or links to any external account or
> private service. See [Anonymous Policy](#anonymous-policy) for details.

## Repository layout

```
GIFT/
├── main.py                       # Single-window entry point
├── core/                         # Library code (env, PPO, GIFT controller, IC/SHAP)
├── configs/                      # YAML configs: config_demo + config_W1..W7 + main
├── scripts/
│   ├── prepare_data.py           # CSV -> pickle preprocessing
│   ├── train_single_window.py    # Train one window, save JSON
│   ├── run_7_windows.py          # Multi-window orchestration (multi-GPU)
│   ├── run_seeds_6windows.sh     # Bash dispatcher: 6 windows × 5 seeds = 30 runs
│   ├── run_baseline.py           # Pure PPO baseline (no LLM features)
│   ├── train_and_compare.py      # GIFT vs baseline training curves
│   └── run_parallel_compare.py   # Parallel comparison driver
├── tests/                        # Integration smoke tests
├── data/                         # User-provided CSV goes here (not shipped)
├── requirements.txt
└── .gitignore
```

## Environment

- Python 3.10+ (3.11 tested).
- PyTorch 2.0+ (CUDA 11.8 or newer recommended; CPU also works for single-window
  runs).
- A multi-GPU machine (4× A100-class) is recommended for the full 7-window × 5-seed
  sweep but is not required to reproduce a single window.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If your platform needs a specific PyTorch wheel (e.g. CUDA 11.8), install
`torch` separately from the official index before running the command above.

## Data Preparation

The codebase expects a single CSV of daily SP500-style prices at
`./data/sp500_prices.csv`. Required columns (case-insensitive, any column order):

| column      | type   | notes                                        |
|-------------|--------|----------------------------------------------|
| `date`      | string | `YYYY-MM-DD`                                 |
| `ticker`    | string | e.g. `TSLA`, `AAPL` (column may be `symbol`) |
| `open`      | float  |                                              |
| `high`      | float  |                                              |
| `low`       | float  |                                              |
| `close`     | float  |                                              |
| `adj_close` | float  | adjusted close (used as the price feature)   |
| `volume`    | float  |                                              |

The default config uses the 5-ticker universe `[TSLA, NFLX, AMZN, MSFT, JNJ]` over
roughly 2018–2024, but any superset works (filtered at load time). The project does
not ship market data; build the CSV with `yfinance` or any equivalent vendor.

Convert the CSV into the date-indexed pickle that `PortfolioEnv` consumes:

```bash
# Place your CSV at ./data/sp500_prices.csv (or pass --csv path/to/file.csv):
python scripts/prepare_data.py
```

The script runs three steps and prints a report:
1. **Validate** the CSV schema (column names) and per-ticker row coverage — fails
   fast with a clear message if anything is missing.
2. **Convert** to `./data/portfolio_5stocks.pkl` (~5 s on a 250 MB CSV).
3. **Verify** the output by loading the pickle and reporting per-ticker date
   coverage.

Custom universe / date window:

```bash
python scripts/prepare_data.py \
    --csv path/to/prices.csv \
    --output data/portfolio_5stocks.pkl \
    --tickers TSLA NFLX AMZN MSFT JNJ \
    --start 2018-01-01 --end 2024-12-31
```

## API Key

The GIFT loop calls an OpenAI-compatible chat completion endpoint. Export your
credentials via environment variables (the code never reads keys from anywhere
else):

```bash
export OPENAI_API_KEY=sk-...
# Optional: point at any OpenAI-compatible endpoint
export OPENAI_BASE_URL=https://api.openai.com/v1
```

The model is set in `configs/*.yaml` (`llm.model`, default `gpt-4o-mini`). If the
key is missing, scripts that need the LLM will raise a clear error; pipeline /
sandbox unit tests do not need a real key.

## Quick Demo

`configs/config_demo.yaml` is a lightweight configuration tuned for smoke testing:
2 iterations × 1 code sample × 5 PPO episodes. End-to-end runtime is about a
minute on a single GPU (or a few minutes on CPU).

```bash
export OPENAI_API_KEY=sk-...
python main.py --config configs/config_demo.yaml --experiment_name demo --seed 42
```

Outputs land under `./results/demo/`. The demo exists to verify the install /
data / API-key configuration end-to-end; it is **not** a replacement for the
main experiments.

## Reproducing Main Experiments

### Single window

```bash
python main.py \
  --config configs/config_W1.yaml \
  --experiment_name W1_seed_42 \
  --seed 42
```

### Full main result — 7 windows × 5 seeds

The shell dispatcher launches 6 windows (W1–W4, W6–W7) × 5 seeds = 30 runs and
picks the least-loaded GPU for each launch (W5 is launched separately or added
back to `WINDOWS`):

```bash
export OPENAI_API_KEY=sk-...
export GPUS="0 1 2 3"          # optional (default: "0 1 2 3")
export MAX_PARALLEL=8          # optional
export RESULTS_DIR=results     # optional
bash scripts/run_seeds_6windows.sh
```

A pure-Python orchestrator for all 7 windows is also provided:

```bash
python scripts/run_7_windows.py
```

### Baselines and training-curve comparison

```bash
# Pure PPO without LLM features (uses the window described by --config):
python scripts/run_baseline.py --config configs/config_W1.yaml --name ppo_baseline_W1

# GIFT vs baseline training-curve comparison across all 7 windows:
python scripts/train_and_compare.py
# Outputs land under ./training_comparison/ (plots + raw_metrics.json).
```

## Expected Outputs

After a single-window run, outputs land under `./results/<experiment_name>/`:

- `iteration_{i}/code.py`        — LLM-generated `revise_state` +
                                   `intrinsic_reward`.
- `iteration_{i}/config.json`    — feature / reward-rule selections used for
                                   this iteration.
- `iteration_{i}/metrics.json`   — per-iteration IC profile, SHAP profile, and
                                   validation Sharpe.
- `best_model.pt`                — actor + critic weights of the best
                                   iteration.
- `final_comparison.json`        — best GIFT run vs PPO baselines (Sharpe,
                                   Sortino, max drawdown, Calmar, total
                                   return, average weights).
- `summary.json`                 — end-to-end iteration history.

For the multi-seed sweep, results land under `./results/W{n}_seed_{s}/...`.
The `train_and_compare.py` script aggregates these into per-window training
curves saved under `./training_comparison/` (plots + `raw_metrics.json`).

## Tests

```bash
python -m pytest tests/
```

The smoke tests cover imports, indicator/registry sanity, `code_sandbox`
validation, and a short PPO update loop. The tests do not call the LLM, so a
dummy `OPENAI_API_KEY` (e.g. `sk-test-dummy`) is sufficient. A
`./data/portfolio_5stocks.pkl` produced by `scripts/prepare_data.py` is required
to exercise the env-dependent cases.

## Reproducibility Notes

- All experiments in the paper use seeds `{42, 123, 456, 789, 1024}`.
- Window definitions live in `configs/config_W{1..7}.yaml` (train / val / test
  periods).
- LLM nondeterminism (`temperature=0.7`) is the dominant source of variance
  across seeds; PPO baselines are exactly reproducible given a seed.

## Anonymous Policy

This repository is a double-blind submission release. All identifying
information has been removed:

- No author names, affiliations, advisors, lab names, or contact addresses.
- No personal email addresses or GitHub usernames.
- No absolute paths referencing personal accounts (`/home/...`, `/Users/...`,
  `C:\Users\...`, etc.).
- No API keys, tokens, private proxies, or internal service URLs in plain text;
  every secret is read from environment variables.
- No WandB / TensorBoard / cloud-logging credentials.

If reviewers identify any residual identifying information, please flag it via
the conference's communication channel; we will rectify in the camera-ready
release. After acceptance, a non-anonymous repository with full citation,
attribution, and contact information will replace this one.

## Citation

```bibtex
@misc{anonymous2026gift,
  title  = {GIFT: LLM-Guided Iterative Feature Tuning for Portfolio
            Optimization with PPO},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Anonymous double-blind submission},
}
```

The bibtex entry above is a placeholder for the double-blind period. The
official citation will be provided in the camera-ready release.
