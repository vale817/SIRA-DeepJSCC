#!/usr/bin/env python3
"""Measure whether SIRA's reliability mapper changes spatial power allocation.

Examples:
  python analyze_r_sensitivity.py --latent_ch 2 --dataset kodak
  python analyze_r_sensitivity.py --latent_ch 2 --dataset div2k --max_images 24
  python analyze_r_sensitivity.py --latent_ch 2 --image data/kodak/kodim05.png
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CHANNEL, LATENT_CH, RESULT_DIR
from models import DEVICE, SIRA_NO_R_METHODS, power_normalize
from visualize_power_map import list_images, load_image_tensor, load_sira


COLORS = {
    'sira_b2_init': '#3ABF99',
    'sira_b2_no_r': '#9B8AC4',
}
LABELS = {
    'sira_b2_init': 'Full SIRA-B2',
    'sira_b2_no_r': 'SIRA-B2 w/o R',
}


def pearson_corr(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() < 1e-12:
        return 1.0 if torch.allclose(a, b, atol=1e-8, rtol=1e-6) else 0.0
    return (torch.dot(a, b) / denom).item()


@torch.no_grad()
def extract_power_map(net, x, snr):
    """Run only encoder + SIRA modules; channel noise cannot affect this result."""
    x = x.to(DEVICE)
    z = power_normalize(net.encoder(x))
    m, _ = net.M(x)

    if net.method in SIRA_NO_R_METHODS:
        z, power_map = net.A(z, m)
        return power_map.detach().float().cpu(), None, None

    snr_tensor = torch.full((x.shape[0],), float(snr), device=DEVICE)
    r_embed, tau = net.R(snr_tensor)
    z, power_map = net.A(z, m, r_embed, tau)
    return (
        power_map.detach().float().cpu(),
        r_embed.detach().float().cpu(),
        tau.detach().float().cpu(),
    )


def choose_paths(args):
    if args.image:
        if not os.path.isfile(args.image):
            raise FileNotFoundError(args.image)
        return [args.image]
    paths = list_images(args.dataset, args.coco_root)
    if not paths:
        raise FileNotFoundError(
            f'No images found for dataset={args.dataset}. '
            'Use --image or check the dataset path.'
        )
    return paths[:args.max_images] if args.max_images else paths


def summarize(rows, methods, snrs, reference_snr):
    summaries = {}
    for method in methods:
        summaries[method] = {}
        for snr in snrs:
            selected = [r for r in rows if r['method'] == method and r['snr'] == snr]
            summaries[method][str(snr)] = {
                'relative_mad_mean': float(np.mean([r['relative_mad'] for r in selected])),
                'relative_mad_std': float(np.std([r['relative_mad'] for r in selected])),
                'correlation_mean': float(np.mean([r['correlation'] for r in selected])),
                'correlation_std': float(np.std([r['correlation'] for r in selected])),
                'power_cv_mean': float(np.mean([r['power_cv'] for r in selected])),
                'power_cv_std': float(np.std([r['power_cv'] for r in selected])),
            }
    return {
        'reference_snr': reference_snr,
        'interpretation': {
            'relative_mad': 'Mean absolute power-map change divided by reference-map mean; 0 means unchanged.',
            'correlation': 'Pearson spatial correlation with the reference power map; 1 means same spatial pattern.',
            'power_cv': 'Spatial coefficient of variation; larger means more concentrated allocation.',
        },
        'methods': summaries,
    }


def write_csv(rows, path):
    fields = [
        'method', 'image', 'snr', 'reference_snr',
        'relative_mad', 'correlation', 'power_cv', 'tau', 'r_embed_l2_vs_ref',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, methods, snrs, output_path):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
    specs = [
        ('relative_mad_mean', 'Relative MAD vs reference', 'Power-map change'),
        ('correlation_mean', 'Correlation vs reference', 'Spatial-pattern similarity'),
        ('power_cv_mean', 'Power-map CV', 'Allocation concentration'),
    ]
    for ax, (key, ylabel, title) in zip(axes, specs):
        for method in methods:
            values = [summary['methods'][method][str(snr)][key] for snr in snrs]
            ax.plot(
                snrs, values, marker='o', linewidth=1.8, markersize=4.8,
                color=COLORS.get(method, None), label=LABELS.get(method, method),
            )
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, color='#BDBDBD', alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(os.path.splitext(output_path)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)


def print_key_result(summary, methods, snrs, reference_snr):
    high_snr = snrs[-1]
    print(f'\nReference SNR: {reference_snr:g} dB; comparison SNR: {high_snr:g} dB')
    for method in methods:
        result = summary['methods'][method][str(high_snr)]
        print(
            f"{LABELS.get(method, method):>18}: "
            f"relative MAD={result['relative_mad_mean']:.6f}, "
            f"corr={result['correlation_mean']:.6f}, "
            f"CV={result['power_cv_mean']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None)
    parser.add_argument('--dataset', default='kodak', choices=['kodak', 'div2k', 'coco'])
    parser.add_argument('--coco_root', default='data/coco')
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--methods', nargs='+',
                        default=['sira_b2_init', 'sira_b2_no_r'])
    parser.add_argument('--channel', default=CHANNEL)
    parser.add_argument('--latent_ch', type=int, default=LATENT_CH)
    parser.add_argument('--snrs', nargs='+', type=float, default=[-2, 0, 2, 5, 10, 15])
    parser.add_argument('--reference_snr', type=float, default=-2)
    parser.add_argument('--result_dir', default=RESULT_DIR)
    args = parser.parse_args()

    if args.reference_snr not in args.snrs:
        raise ValueError('--reference_snr must be included in --snrs')
    os.makedirs(args.result_dir, exist_ok=True)
    paths = choose_paths(args)
    print(f'Device: {DEVICE}')
    print(f'Images: {len(paths)}')

    rows = []
    r_outputs = {}
    for method in args.methods:
        print(f'Analyzing {method}...')
        net = load_sira(method, args.channel, args.latent_ch)
        for image_path in paths:
            x, _ = load_image_tensor(image_path)
            outputs = {
                snr: extract_power_map(net, x, snr)
                for snr in args.snrs
            }
            ref_map, ref_embed, _ = outputs[args.reference_snr]
            for snr in args.snrs:
                power_map, r_embed, tau = outputs[snr]
                mean_ref = ref_map.abs().mean().item()
                relative_mad = (power_map - ref_map).abs().mean().item() / (mean_ref + 1e-12)
                cv = power_map.std().item() / (power_map.mean().item() + 1e-12)
                embed_l2 = None
                if r_embed is not None:
                    embed_l2 = torch.linalg.vector_norm(r_embed - ref_embed).item()
                    r_outputs[str(snr)] = {
                        'tau': tau.mean().item(),
                        'r_embed_l2_vs_reference': embed_l2,
                    }
                rows.append({
                    'method': method,
                    'image': os.path.basename(image_path),
                    'snr': snr,
                    'reference_snr': args.reference_snr,
                    'relative_mad': relative_mad,
                    'correlation': pearson_corr(power_map, ref_map),
                    'power_cv': cv,
                    'tau': None if tau is None else tau.mean().item(),
                    'r_embed_l2_vs_ref': embed_l2,
                })

    summary = summarize(rows, args.methods, args.snrs, args.reference_snr)
    summary['dataset'] = args.dataset if not args.image else args.image
    summary['num_images'] = len(paths)
    summary['full_model_r_outputs'] = r_outputs

    prefix = os.path.join(
        args.result_dir, f'r_sensitivity_{args.dataset}_{args.channel}_c{args.latent_ch}'
    )
    write_csv(rows, prefix + '.csv')
    with open(prefix + '.json', 'w') as f:
        json.dump(summary, f, indent=2)
    plot_summary(summary, args.methods, args.snrs, prefix + '.png')
    print_key_result(summary, args.methods, args.snrs, args.reference_snr)
    print(f'\nSaved: {prefix}.csv')
    print(f'Saved: {prefix}.json')
    print(f'Saved: {prefix}.png / .pdf')


if __name__ == '__main__':
    main()
