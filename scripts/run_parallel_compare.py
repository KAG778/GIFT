"""Run parallel training for all windows."""
import sys
import os
from pathlib import Path
from multiprocessing import Process, Queue

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'core'))
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))

from train_and_compare import (
    run_gift_training, run_baseline_training, load_best_gift_code,
    plot_comparison, generate_report, find_convergence_episode,
    compute_reward_stability, WINDOWS, MAX_EPISODES
)
import json
import numpy as np


def run_window(window, seed, config_path, result_queue):
    """Run both GIFT and baseline training for one window."""
    result = {'gift': None, 'baseline': None}
    try:
        print(f"[{window}] Starting GIFT-PPO training...")
        gift_metrics = run_gift_training(config_path, window, seed)
        if gift_metrics is not None:
            result['gift'] = gift_metrics
            print(f"[{window}] GIFT-PPO done: Sharpe={gift_metrics['final_sharpe']:.3f}")
    except Exception as e:
        print(f"[{window}] GIFT training failed: {e}")
        import traceback; traceback.print_exc()

    try:
        print(f"[{window}] Starting Baseline-PPO training...")
        base_metrics = run_baseline_training(config_path, seed)
        if base_metrics is not None:
            result['baseline'] = base_metrics
            print(f"[{window}] Baseline-PPO done: Sharpe={base_metrics['final_sharpe']:.3f}")
    except Exception as e:
        print(f"[{window}] Baseline training failed: {e}")
        import traceback; traceback.print_exc()

    result_queue.put((window, result))


def main():
    seed = 123
    all_results = {}

    processes = []
    result_queue = Queue()

    for window in WINDOWS:
        config_path = str(PROJECT_DIR / 'configs' / f'config_{window}.yaml')
        if not Path(config_path).exists():
            print(f"Config not found: {config_path}, skipping {window}")
            continue
        p = Process(target=run_window, args=(window, seed, config_path, result_queue))
        processes.append((window, p))
        p.start()
        print(f"Started {window} (PID={p.pid})")

    for window, p in processes:
        p.join()
        print(f"{window} finished")

    while not result_queue.empty():
        window, result = result_queue.get()
        all_results[window] = result

    # Save raw metrics
    output_dir = PROJECT_DIR / 'training_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'raw_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(all_results, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(all_results, output_dir)

    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
