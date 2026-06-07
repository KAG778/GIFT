# GIFT: LLM-Guided State-Reward Interface for Financial Reinforcement Learning

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [API Key and LLM Endpoint](#api-key-and-llm-endpoint)
- [Quick Demo](#quick-demo)
- [Reproducing Main Experiments](#reproducing-main-experiments)
- [Expected Outputs](#expected-outputs)
- [Tests](#tests)
- [Reproducibility Notes](#reproducibility-notes)
- [Anonymous Policy](#anonymous-policy)
- [Citation](#citation)

## Overview

GIFT is a framework in which an LLM acts as the **state-reward interface** for a
downstream RL agent solving financial decision-making tasks. Concretely, the LLM
generates two Python functions per iteration:

- `revise_state`: augments the raw market observation into a richer state
  representation;
- `intrinsic_reward`: emits an auxiliary reward signal aligned with the task.

A PPO agent trains on the augmented state with the augmented reward; per-feature
IC and SHAP diagnostics are computed on the trained critic and fed back to the
LLM as natural-language COT feedback, closing the loop for the next iteration.

This repository implements GIFT for **portfolio optimization with PPO** as the
main experimental instantiation. Each portfolio panel holds five US equities
plus a cash position under a long-only simplex constraint, evaluated over six
rolling windows from 2019 to 2024, with a mean-variance reward augmented by the
LLM-generated intrinsic reward and rule-based shaping. The full evaluation
spans **six panels** — three single-sector (Technology, Healthcare, Energy),
two mixed (Light Mix, Heavy Mix), and one Industrials panel — so the main
result is a **6 panels × 6 windows** grid. See
[Reproducing Main Experiments](#reproducing-main-experiments).

## Repository layout

```
GIFT/
├── main.py                       # Entry point: runs one config (panel/window)
├── core/                         # Library code (env, PPO, GIFT controller, IC/SHAP)
├── configs/
│   ├── config.yaml               # Base/default config (Light Mix panel)
│   ├── config_W1..W6.yaml        # Per-window configs for the Light Mix panel
│   ├── windows.yaml              # The 6 rolling windows (shared by all panels)
│   └── panels/                   # One config per portfolio panel (6 files)
├── scripts/
│   ├── prepare_data.py           # CSV -> pickle preprocessing (per panel)
│   ├── run_panels.py             # 6 panels × 6 windows orchestration
│   ├── train_single_window.py    # Train one window, save JSON
│   ├── run_6_windows.py          # Light Mix panel, 6-window orchestration
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

- Python 3.10+ (tested with Python 3.10).
- PyTorch 2.x — **match the wheel to your GPU driver**. Tested with
  `torch 2.5.1` (CUDA 12.4) on an NVIDIA A100. A wheel built for a newer CUDA
  than your driver supports will silently fall back to CPU; if `torch` reports
  `cuda.is_available() == False`, install a `torch` build matching your driver
  (see *Installation*). CPU also works for the demo / a single run.
- A GPU is recommended for the full 6 panels × 6 windows grid but is not
  required to run the demo or a single panel/window.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If your platform needs a specific PyTorch wheel (e.g. CUDA 11.8), install
`torch` separately from the official index before running the command above.

## Data Preparation

The codebase expects a single CSV of daily SP500-style prices at
`./data/sp500_prices.csv`.

### Where to get the data

We use the publicly released **FINSABER price dataset** (an external,
third-party benchmark dataset — not produced by the authors of this
submission). It covers US equity prices from 2000 to 2024 (including
delisted tickers), hosted on Hugging Face.

**One-line download (recommended)**:

```bash
bash scripts/download_data.sh
```

This script picks `wget` or `curl` automatically, skips the download if the
file already exists with a sane size, validates the byte count after
download, and prints the next command to run. Pass `--force` to re-download
unconditionally.

**Manual download** if you prefer:

```bash
mkdir -p data
wget -O data/sp500_prices.csv \
    https://huggingface.co/datasets/waylonli/FINSABER-data/resolve/main/data/price/all_sp500_prices_2000_2024_delisted_include.csv
# or:
# curl -L -o data/sp500_prices.csv \
#     https://huggingface.co/datasets/waylonli/FINSABER-data/resolve/main/data/price/all_sp500_prices_2000_2024_delisted_include.csv
```

Please consult the dataset's Hugging Face page for license and citation
information. The expected CSV schema (case-insensitive, any column order):

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

The stock list for each run comes entirely from `config['data']['tickers']`;
no ticker symbols are hardcoded in the library. The single downloaded CSV is a
superset that covers all six panels — non-target tickers are filtered out at
load time. If you prefer your own data source, just produce a CSV matching the
schema above and drop it at `./data/sp500_prices.csv`.

### Convert CSV to per-panel pickles

`scripts/prepare_data.py` extracts one panel's tickers from the CSV into a
date-indexed pickle. It (1) **validates** the CSV schema and per-ticker
coverage, (2) **converts** to the requested pickle (~5 s on a 250 MB CSV), and
(3) **verifies** the output by reloading and reporting per-ticker date coverage.

Generate the pickle for every panel in one loop:

```bash
for p in "technology:AAPL MSFT NVDA GOOGL META" \
         "healthcare:JNJ LLY UNH MRK PFE" \
         "energy:XOM CVX COP SLB EOG" \
         "light_mix:TSLA NFLX AMZN MSFT JNJ" \
         "heavy_mix:TSLA NVDA XOM CAT JNJ" \
         "industrials:BA CAT UNP LMT WM"; do
  name=${p%%:*}; ticks=${p#*:}
  python scripts/prepare_data.py --csv data/sp500_prices.csv \
      --tickers $ticks --output data/portfolio_${name}.pkl
done
```

Each panel config (`configs/panels/<panel>.yaml`) points its `data.pickle_file`
at the matching `data/portfolio_<panel>.pkl`. To prepare a single panel (or a
custom universe / date window):

```bash
python scripts/prepare_data.py \
    --csv data/sp500_prices.csv \
    --output data/portfolio_technology.pkl \
    --tickers AAPL MSFT NVDA GOOGL META \
    --start 2018-01-01 --end 2024-12-31
```

## API Key and LLM Endpoint

The GIFT loop calls an OpenAI-compatible chat-completion endpoint.

- **API key** is read from the environment first (a key written in a config is
  only a fallback):

  ```bash
  export OPENAI_API_KEY=sk-...
  ```

- **Endpoint and model** are set per config under `llm`: `llm.base_url`
  (default `https://api.openai.com/v1`) and `llm.model` (default
  `gpt-4o-mini`). To use an OpenAI-compatible proxy, edit `llm.base_url` in the
  config you run — the endpoint is taken from the config, not from an
  environment variable.

If the key is missing or invalid, scripts that call the LLM raise a clear error;
the pipeline / sandbox unit tests do not need a real key.

## Quick Demo

`configs/config_demo.yaml` is a lightweight configuration tuned for smoke testing:
2 iterations × 1 code sample × 5 PPO episodes. End-to-end runtime is about a
minute on a single GPU (or a few minutes on CPU). It uses the Light Mix panel,
so it needs `data/portfolio_light_mix.pkl` from *Data Preparation* and a valid
API key:

```bash
export OPENAI_API_KEY=sk-...
# one-time data prep for the demo panel (skip if already done):
python scripts/prepare_data.py --csv data/sp500_prices.csv \
    --tickers TSLA NFLX AMZN MSFT JNJ --output data/portfolio_light_mix.pkl
python main.py --config configs/config_demo.yaml --experiment_name demo --seed 42
```

Outputs land under `./results/demo/`. The demo exists to verify the install /
data / API-key configuration end-to-end; it is **not** a replacement for the
main experiments.

## Reproducing Main Experiments

The main result is a **6 panels × 6 windows** grid. Each panel is defined by a
config in `configs/panels/`; the six rolling windows are defined once in
`configs/windows.yaml`. The runner composes panel × window into a generated
config and launches `main.py` for each.

### All panels × all windows (main result)

```bash
export OPENAI_API_KEY=sk-...
# All 6 panels × 6 windows = 36 runs (serial by default, GPU 0):
python scripts/run_panels.py

# A subset (specific panels and/or windows):
python scripts/run_panels.py --panels technology healthcare --windows W1 W2 --gpu 0

# Generate the merged configs without launching (inspect them first):
python scripts/run_panels.py --dry-run
```

Generated per-run configs are written under `configs/_generated/`
(`<panel>_<window>.yaml`) and results under `./results/<panel>_<window>/`.
`run_panels.py` is serial by default; to use multiple GPUs, launch disjoint
panel/window subsets with different `--gpu` values.

### Single run (one panel, one window)

Compose one config by hand, or reuse a generated one:

```bash
python scripts/run_panels.py --panels technology --windows W1 --dry-run
python main.py \
  --config configs/_generated/technology_W1.yaml \
  --experiment_name technology_W1 \
  --seed 42
```

### Single-panel windows (Light Mix)

For convenience, `configs/config_W1..W6.yaml` run the Light Mix panel directly,
one window per config, without the panel composer. They expect
`data/portfolio_5stocks.pkl` (the Light Mix universe), produced with:

```bash
python scripts/prepare_data.py --csv data/sp500_prices.csv \
    --tickers TSLA NFLX AMZN MSFT JNJ --output data/portfolio_5stocks.pkl
python main.py \
  --config configs/config_W1.yaml \
  --experiment_name W1_seed_42 \
  --seed 42
```

### Multi-seed variance sweep (Light Mix panel)

`scripts/run_panels.py` runs a single seed per (panel, window) cell. To measure
seed variance, `scripts/run_seeds_6windows.sh` sweeps the Light Mix panel over 6
windows (W1–W6) × 5 seeds = 30 runs, picking the least-loaded GPU for each launch
(to sweep seeds on another panel, point its `config_W*`-style configs at that
panel's tickers/pickle, or add a `--seed` loop around `run_panels.py`):

```bash
export OPENAI_API_KEY=sk-...
export GPUS="0 1 2 3"          # optional (default: "0 1 2 3")
export MAX_PARALLEL=8          # optional
export RESULTS_DIR=results     # optional
bash scripts/run_seeds_6windows.sh
```

A pure-Python orchestrator for the same 6 windows is also provided:

```bash
python scripts/run_6_windows.py
```

### Baselines and training-curve comparison

```bash
# Pure PPO without LLM features (uses the window described by --config):
python scripts/run_baseline.py --config configs/config_W1.yaml --name ppo_baseline_W1

# GIFT vs baseline training-curve comparison across all 6 windows:
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
validation, and a short PPO update loop. `tests/test_multi_panel.py` is a
regression guard for the config-driven stock list: it builds the environment
and portfolio features for a non-default panel from synthetic data and asserts
the panel's tickers flow end-to-end. The tests do not call the LLM, so a dummy
`OPENAI_API_KEY` (e.g. `sk-test-dummy`) is sufficient; `test_multi_panel.py`
needs no key and no downloaded data. A `./data/portfolio_<panel>.pkl` produced
by `scripts/prepare_data.py` is required to exercise the env-dependent cases of
the broader integration test.

## Reproducibility Notes

- All experiments in the paper use seeds `{42, 123, 456, 789, 1024}`.
- LLM nondeterminism (`temperature=0.7`) is the dominant source of variance
  across seeds; PPO baselines are exactly reproducible given a seed.
- The 6 rolling windows use an 18-month train + 12-month test design with a
  6-month step. The val period equals the test period (no separate hold-out
  for selection — the best iteration is chosen by training-period validation
  inside `GIFTController.run()`):

  | Window | Train period            | Test period             |
  |--------|-------------------------|-------------------------|
  | W1     | 2019-01-01 .. 2020-06-30 | 2020-07-01 .. 2021-06-30 |
  | W2     | 2019-07-01 .. 2020-12-31 | 2021-01-01 .. 2021-12-31 |
  | W3     | 2020-01-01 .. 2021-06-30 | 2021-07-01 .. 2022-06-30 |
  | W4     | 2020-07-01 .. 2021-12-31 | 2022-01-01 .. 2022-12-31 |
  | W5     | 2021-01-01 .. 2022-06-30 | 2022-07-01 .. 2023-06-30 |
  | W6     | 2021-07-01 .. 2022-12-31 | 2023-01-01 .. 2023-12-31 |

  All 6 windows share an identical configuration except for these date
  ranges; they are defined once in `configs/windows.yaml` and applied to every
  panel by `scripts/run_panels.py`.

- **Design invariant — five equities + cash.** Every panel holds exactly
  **five risky assets plus one cash position**, i.e. a **6-dimensional**,
  long-only weight vector (cash is the last component). The five tickers are
  fully **config-driven** (`config['data']['tickers']`) and no ticker symbol is
  hardcoded in the library — swapping a panel means editing only the config.
  The *count*, however, is a fixed design constant across all panels and
  experiments: the few positional constants in the code (a 6-dim weight vector
  whose first five entries are the stocks and the sixth is cash) intentionally
  encode this five-plus-cash invariant and are not meant to vary per run.

- Each panel config additionally declares `growth` / `defensive` ticker
  groups, used only by the optional `sector_exposure` portfolio feature; the
  groups partition the five tickers and carry no other semantics.

  | Panel        | Tickers                        | Growth                  | Defensive        |
  |--------------|--------------------------------|-------------------------|------------------|
  | Technology   | AAPL, MSFT, NVDA, GOOGL, META  | AAPL, NVDA, GOOGL, META | MSFT             |
  | Healthcare   | JNJ, LLY, UNH, MRK, PFE        | LLY, UNH                | JNJ, MRK, PFE    |
  | Energy       | XOM, CVX, COP, SLB, EOG        | COP, EOG, SLB           | XOM, CVX         |
  | Light Mix    | TSLA, NFLX, AMZN, MSFT, JNJ    | TSLA, NFLX, AMZN, MSFT  | JNJ              |
  | Heavy Mix    | TSLA, NVDA, XOM, CAT, JNJ      | TSLA, NVDA, CAT         | XOM, JNJ         |
  | Industrials  | BA, CAT, UNP, LMT, WM          | BA, CAT                 | UNP, LMT, WM     |

  All panels share an identical configuration except for the `data` block; see
  `configs/panels/<panel>.yaml`.

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
  title  = {GIFT: LLM-Guided State-Reward Interface for Financial
            Reinforcement Learning},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Anonymous double-blind submission},
}
```

The bibtex entry above is a placeholder for the double-blind period. The
official citation will be provided in the camera-ready release.
