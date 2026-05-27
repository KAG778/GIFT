"""
Main entry point for portfolio optimization with GIFT.

Usage:
    python main.py --config configs/config.yaml --experiment_name my_experiment

Pipeline:
    1. Load the YAML config file.
    2. Construct a GIFTController.
    3. Run the full iterative optimization loop.
    4. Emit final test results and baseline comparison.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from gift_controller import GIFTController


def main():
    parser = argparse.ArgumentParser(description='GIFT Portfolio Optimization')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--experiment_name', type=str, default='gift_portfolio',
                        help='Experiment name for results directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create controller and run
    experiment_dir = str(Path('results') / args.experiment_name)
    controller = GIFTController(config, experiment_dir, seed=args.seed)
    controller.run()


if __name__ == '__main__':
    main()
