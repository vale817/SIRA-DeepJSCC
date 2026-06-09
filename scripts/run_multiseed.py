#!/usr/bin/env python3
"""Run SIRA experiments across multiple random seeds and aggregate metrics."""
import argparse
import json
import math
import os
import subprocess
import sys

from sira.config import (
    BATCH_SIZE, CHANNEL, CKPT_DIR, EPOCHS_B1, EPOCHS_B2, EPOCHS_SIRA,
    IMPORTANCE_MODE, LATENT_CH, RESULT_DIR, SEED, ALLOCATION_MODE,
)
from sira.models import SIRA_METHODS


DEFAULT_METHODS = ['cnn', 'semantic', 'sira_b1_init', 'sira_b2_init']
DATASET_FILES = {
    'div2k': 'div2k_{channel}_c{latent_ch}.json',
    'kodak': 'kodak_{channel}_c{latent_ch}.json',
    'coco': 'coco_object_{channel}_c{latent_ch}.json',
}


def run_command(cmd, dry_run=False):
    print('\n$ ' + ' '.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.check_call(cmd)


def seed_dir(root, seed):
    return os.path.join(root, f'seed_{seed}')


def mean_std(values):
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return round(mean, 6), round(math.sqrt(var), 6)


def aggregate_dataset(result_root, seeds, dataset, channel, latent_ch):
    filename = DATASET_FILES[dataset].format(channel=channel, latent_ch=latent_ch)
    loaded = []
    for seed in seeds:
        path = os.path.join(seed_dir(result_root, seed), filename)
        if not os.path.isfile(path):
            print(f'skip aggregate missing {path}')
            continue
        with open(path) as f:
            loaded.append((seed, json.load(f)))

    if not loaded:
        return None

    first = loaded[0][1]
    methods = sorted(first['methods'].keys())
    summary = {
        'dataset': first.get('dataset', dataset),
        'allocation_mode': first.get('allocation_mode'),
        'seeds': [seed for seed, _ in loaded],
        'snr_sweep': first.get('snr_sweep'),
        'methods': {},
    }

    for method in methods:
        metrics = sorted(first['methods'][method].keys())
        summary['methods'][method] = {}
        for metric in metrics:
            metric_values = []
            for _, data in loaded:
                if method not in data['methods'] or metric not in data['methods'][method]:
                    metric_values = []
                    break
                metric_values.append(data['methods'][method][metric])
            if not metric_values or not isinstance(metric_values[0], list):
                continue

            n_points = len(metric_values[0])
            means, stds = [], []
            for idx in range(n_points):
                vals = [float(seed_values[idx]) for seed_values in metric_values]
                mean, std = mean_std(vals)
                means.append(mean)
                stds.append(std)
            summary['methods'][method][metric] = {
                'mean': means,
                'std': stds,
            }

    out = os.path.join(result_root, f'summary_{dataset}_{channel}_c{latent_ch}.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'aggregate saved -> {out}')
    return summary


def aggregate_results(result_root, seeds, channel, latent_ch, include_coco=False):
    datasets = ['div2k', 'kodak']
    if include_coco:
        datasets.append('coco')
    for dataset in datasets:
        aggregate_dataset(result_root, seeds, dataset, channel, latent_ch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[SEED, SEED + 1, SEED + 2])
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS,
                        choices=['cnn', 'semantic'] + list(SIRA_METHODS))
    parser.add_argument('--eval_methods', nargs='+', default=None,
                        choices=['cnn', 'semantic'] + list(SIRA_METHODS))
    parser.add_argument('--channel', default=CHANNEL)
    parser.add_argument('--latent_ch', type=int, default=LATENT_CH)
    parser.add_argument('--ckpt_root', default=os.path.join(CKPT_DIR, 'multiseed'))
    parser.add_argument('--result_root', default=os.path.join(RESULT_DIR, 'multiseed'))
    parser.add_argument('--importance_mode', default=IMPORTANCE_MODE,
                        choices=['edge', 'dino'])
    parser.add_argument('--allocation_mode', default=ALLOCATION_MODE,
                        choices=['hard', 'soft'])
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs_b1', type=int, default=EPOCHS_B1)
    parser.add_argument('--epochs_b2', type=int, default=EPOCHS_B2)
    parser.add_argument('--epochs_sira', type=int, default=EPOCHS_SIRA)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--skip_div2k', action='store_true')
    parser.add_argument('--skip_kodak', action='store_true')
    parser.add_argument('--eval_coco', action='store_true')
    parser.add_argument('--coco_root', default='data/coco')
    parser.add_argument('--coco_max_images', type=int, default=None)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    eval_methods = args.eval_methods or args.methods
    os.makedirs(args.ckpt_root, exist_ok=True)
    os.makedirs(args.result_root, exist_ok=True)

    for seed in args.seeds:
        ckpt_dir = seed_dir(args.ckpt_root, seed)
        result_dir = seed_dir(args.result_root, seed)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        if not args.skip_train:
            train_cmd = [
                sys.executable, '-m', 'scripts.train',
                '--methods', *args.methods,
                '--channel', args.channel,
                '--latent_ch', str(args.latent_ch),
                '--epochs_b1', str(args.epochs_b1),
                '--epochs_b2', str(args.epochs_b2),
                '--epochs_sira', str(args.epochs_sira),
                '--batch_size', str(args.batch_size),
                '--lr', str(args.lr),
                '--seed', str(seed),
                '--ckpt_dir', ckpt_dir,
                '--importance_mode', args.importance_mode,
                '--allocation_mode', args.allocation_mode,
            ]
            if args.force:
                train_cmd.append('--force')
            run_command(train_cmd, dry_run=args.dry_run)

        if not args.skip_eval:
            eval_cmd = [
                sys.executable, '-m', 'scripts.eval',
                '--methods', *eval_methods,
                '--channel', args.channel,
                '--latent_ch', str(args.latent_ch),
                '--batch_size', str(args.batch_size),
                '--seed', str(seed),
                '--ckpt_dir', ckpt_dir,
                '--result_dir', result_dir,
                '--importance_mode', args.importance_mode,
                '--allocation_mode', args.allocation_mode,
            ]
            if args.skip_div2k:
                eval_cmd.append('--skip_div2k')
            if args.skip_kodak:
                eval_cmd.append('--skip_kodak')
            run_command(eval_cmd, dry_run=args.dry_run)

        if args.eval_coco:
            coco_cmd = [
                sys.executable, '-m', 'scripts.eval_coco',
                '--methods', *eval_methods,
                '--channel', args.channel,
                '--latent_ch', str(args.latent_ch),
                '--seed', str(seed),
                '--ckpt_dir', ckpt_dir,
                '--result_dir', result_dir,
                '--coco_root', args.coco_root,
                '--allocation_mode', args.allocation_mode,
            ]
            if args.coco_max_images is not None:
                coco_cmd.extend(['--max_images', str(args.coco_max_images)])
            run_command(coco_cmd, dry_run=args.dry_run)

    if not args.dry_run and not args.skip_eval:
        aggregate_results(
            args.result_root,
            args.seeds,
            args.channel,
            args.latent_ch,
            include_coco=args.eval_coco,
        )


if __name__ == '__main__':
    main()
