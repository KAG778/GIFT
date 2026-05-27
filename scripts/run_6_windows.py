#!/usr/bin/env python3
"""
Parallel runner for the 6 rolling windows (W1-W6). Results land under
``./results/``. The default ``WINDOW_GPU`` mapping pins two windows per GPU
across 3 GPUs; edit the list below to match your hardware layout.
"""
import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

script_dir = Path(__file__).parent
project_dir = script_dir.parent
os.chdir(project_dir)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = project_dir / 'results'

# (window, gpu_id) pairs — spread across 3 GPUs
WINDOW_GPU = [
    ('W1', 0), ('W2', 0),
    ('W3', 1), ('W4', 1),
    ('W5', 2), ('W6', 2),
]


def run_window(window: str, gpu_id: int):
    """Launch one window experiment pinned to a specific GPU."""
    config_path = project_dir / 'configs' / f'config_{window}.yaml'
    log_path = RESULTS_DIR / f'{window}.log'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        sys.executable,
        str(project_dir / 'main.py'),
        '--config', str(config_path),
        '--experiment_name', window,
    ]

    logger.info(f"Starting {window} on GPU {gpu_id}")

    with open(log_path, 'w') as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    return proc


def main():
    logger.info(f"Launching 6 windows in parallel — {TIMESTAMP}")
    logger.info(f"Results dir: {RESULTS_DIR}")

    procs = {}
    for w, gpu in WINDOW_GPU:
        procs[w] = run_window(w, gpu)

    logger.info("All 7 processes launched. Waiting...")

    for w, proc in procs.items():
        rc = proc.wait()
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        logger.info(f"  {w}: {status}")

    logger.info("All windows finished.")


if __name__ == '__main__':
    main()
