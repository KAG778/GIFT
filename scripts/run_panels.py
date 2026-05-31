#!/usr/bin/env python3
"""Run panel x window experiments. Generates a merged config per (panel, window)
and launches main.py. main.py itself is unchanged."""
import argparse, os, subprocess, sys, copy
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parent.parent
PANELS_DIR = PROJECT / 'configs' / 'panels'
WINDOWS_FILE = PROJECT / 'configs' / 'windows.yaml'
GEN_DIR = PROJECT / 'configs' / '_generated'


def build_config(panel_path: Path, window: dict) -> Path:
    cfg = yaml.safe_load(open(panel_path))
    cfg = copy.deepcopy(cfg)
    cfg.setdefault('experiment', {})
    cfg['experiment']['train_period'] = window['train_period']
    cfg['experiment']['val_period'] = window['val_period']
    cfg['experiment']['test_period'] = window['test_period']
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_DIR / f"{panel_path.stem}_{window['name']}.yaml"
    yaml.safe_dump(cfg, open(out, 'w'), sort_keys=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panels', nargs='+', default=None,
                    help='panel stems (default: all in configs/panels)')
    ap.add_argument('--windows', nargs='+', default=None,
                    help='window names e.g. W1 W2 (default: all)')
    ap.add_argument('--gpu', type=int, default=0, help='CUDA device id')
    ap.add_argument('--dry-run', action='store_true',
                    help='only generate configs, do not launch')
    args = ap.parse_args()

    windows = yaml.safe_load(open(WINDOWS_FILE))['windows']
    if args.windows:
        windows = [w for w in windows if w['name'] in args.windows]
    panel_paths = ([PANELS_DIR / f'{p}.yaml' for p in args.panels]
                   if args.panels else sorted(PANELS_DIR.glob('*.yaml')))

    for panel_path in panel_paths:
        for w in windows:
            gen = build_config(panel_path, w)
            name = f"{panel_path.stem}_{w['name']}"
            print(f"[{'gen' if args.dry_run else 'run'}] {name} -> {gen}")
            if args.dry_run:
                continue
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            subprocess.run([sys.executable, str(PROJECT / 'main.py'),
                            '--config', str(gen), '--experiment_name', name],
                           env=env, check=False)


if __name__ == '__main__':
    main()
