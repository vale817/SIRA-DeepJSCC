#!/usr/bin/env python3
# ============================================================
# plot_low_snr.py
# Focused low-SNR summary for SIRA experiments.
#
# Usage:
#   python plot_low_snr.py --latent_ch 2
#   python plot_low_snr.py --latent_ch 2 --snrs -2 0 2
# ============================================================
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import RESULT_DIR, CHANNEL, LATENT_CH, METHOD_NAMES, METHOD_STYLE


DATASETS = [
    ('div2k', 'DIV2K Val', 'div2k_{channel}_c{latent_ch}.json', 'roi_psnr'),
    ('kodak', 'Kodak-24', 'kodak_{channel}_c{latent_ch}.json', 'roi_psnr'),
    ('coco', 'COCO Object ROI', 'coco_object_{channel}_c{latent_ch}.json', 'object_roi_psnr'),
]

PUB_STYLE = {
    # Soft journal-style palette inspired by blue/teal/gold scientific figures.
    'cnn': {
        'label': 'CNN-DeepJSCC',
        'color': '#6E7781',
        'marker': 'o',
        'linestyle': '--',
        'hatch': '',
    },
    'semantic': {
        'label': 'Semantic-weighted',
        'color': '#2C91E0',
        'marker': '^',
        'linestyle': ':',
        'hatch': '',
    },
    'sira': {
        'label': 'SIRA-B1-init',
        'color': '#F0A73A',
        'marker': 'D',
        'linestyle': '-',
        'hatch': '',
    },
    'sira_b1_init': {
        'label': 'SIRA-B1-init',
        'color': '#F0A73A',
        'marker': 'D',
        'linestyle': '-',
        'hatch': '',
    },
    'sira_b2_init': {
        'label': 'SIRA-B2-init',
        'color': '#3ABF99',
        'marker': 's',
        'linestyle': '-.',
        'hatch': '',
    },
    'sira_b2_no_r': {
        'label': 'SIRA-B2 w/o R',
        'color': '#9B8AC4',
        'marker': 'v',
        'linestyle': '--',
        'hatch': '',
    },
}


def set_publication_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'figure.titlesize': 11,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'lines.markersize': 5.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
    })


def pub_label(method):
    return PUB_STYLE.get(method, {}).get('label', METHOD_NAMES.get(method, method))


def save_figure(fig, png_path):
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.04)
    pdf_path = os.path.splitext(png_path)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    print(f'saved -> {png_path}')
    print(f'saved -> {pdf_path}')


def load_result(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if 'methods' in data:
        return data['methods'], data.get('snr_sweep')
    raise ValueError(f'Unsupported result JSON format: {path}')


def available_results(result_dir, channel, latent_ch):
    found = []
    for key, title, pattern, roi_key in DATASETS:
        path = os.path.join(
            result_dir,
            pattern.format(channel=channel, latent_ch=latent_ch),
        )
        if os.path.isfile(path):
            found.append((key, title, path, roi_key))
    return found


def method_label(method):
    return METHOD_NAMES.get(method, method)


def collect_rows(result_specs, low_snrs):
    rows = []
    for dataset_key, dataset_title, path, roi_key in result_specs:
        methods, snr_sweep = load_result(path)
        snr_sweep = list(snr_sweep or methods['cnn']['snr'])

        for snr in low_snrs:
            if snr not in snr_sweep:
                continue
            idx = snr_sweep.index(snr)
            b1_psnr = methods['cnn']['psnr'][idx]
            b1_roi = methods['cnn'][roi_key][idx]

            for method, values in methods.items():
                row = {
                    'dataset': dataset_title,
                    'snr': snr,
                    'method': method,
                    'method_name': method_label(method),
                    'psnr': values['psnr'][idx],
                    'ssim': values['ssim'][idx],
                    'lpips': values['lpips'][idx],
                    'roi_metric': roi_key,
                    'roi_psnr': values[roi_key][idx],
                    'delta_psnr_vs_b1': values['psnr'][idx] - b1_psnr,
                    'delta_roi_vs_b1': values[roi_key][idx] - b1_roi,
                }
                rows.append(row)
    return rows


def write_csv(rows, path):
    fields = [
        'dataset', 'snr', 'method', 'method_name',
        'psnr', 'ssim', 'lpips', 'roi_metric', 'roi_psnr',
        'delta_psnr_vs_b1', 'delta_roi_vs_b1',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x):
    return f'{x:.3f}'


def write_markdown(rows, path):
    with open(path, 'w') as f:
        f.write('| Dataset | SNR | Method | PSNR | ROI PSNR | Delta PSNR vs B1 | Delta ROI vs B1 |\n')
        f.write('|---|---:|---|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(
                f"| {r['dataset']} | {r['snr']} | {r['method_name']} | "
                f"{fmt(r['psnr'])} | {fmt(r['roi_psnr'])} | "
                f"{fmt(r['delta_psnr_vs_b1'])} | {fmt(r['delta_roi_vs_b1'])} |\n"
            )


def print_low_snr_table(rows):
    print('\nLow-SNR summary (-2/0/2 dB):')
    print(f"{'Dataset':<16} {'SNR':>4} {'Method':<24} {'PSNR':>8} {'ROI':>8} {'dPSNR':>8} {'dROI':>8}")
    print('-' * 84)
    for r in rows:
        print(
            f"{r['dataset']:<16} {r['snr']:>4} {r['method_name']:<24} "
            f"{r['psnr']:>8.3f} {r['roi_psnr']:>8.3f} "
            f"{r['delta_psnr_vs_b1']:>8.3f} {r['delta_roi_vs_b1']:>8.3f}"
        )


def get_gain_matrix(rows, dataset_title, metric, methods, low_snrs):
    matrix = np.full((len(methods), len(low_snrs)), np.nan)
    for i, method in enumerate(methods):
        for j, snr in enumerate(low_snrs):
            matches = [
                r for r in rows
                if r['dataset'] == dataset_title and r['method'] == method and r['snr'] == snr
            ]
            if matches:
                matrix[i, j] = matches[0][metric]
    return matrix


def plot_gain_figure(rows, result_specs, low_snrs, path):
    set_publication_style()
    present_methods = {r['method'] for r in rows}
    compare_methods = [
        m for m in ('semantic', 'sira', 'sira_b1_init', 'sira_b2_init', 'sira_b2_no_r')
        if m in present_methods
    ]
    n_datasets = len(result_specs)
    fig, axes = plt.subplots(
        n_datasets,
        2,
        figsize=(7.4, max(2.25 * n_datasets, 3.4)),
        squeeze=False,
        constrained_layout=False,
    )

    width = min(0.8 / max(len(compare_methods), 1), 0.30)
    x = np.arange(len(low_snrs))
    handles = []
    labels = []

    for row_idx, (_, dataset_title, _, _) in enumerate(result_specs):
        for col_idx, (metric, ylabel, title) in enumerate([
            ('delta_psnr_vs_b1', 'PSNR gain vs B1 (dB)', 'Global PSNR Gain'),
            ('delta_roi_vs_b1', 'ROI gain vs B1 (dB)', 'ROI PSNR Gain'),
        ]):
            ax = axes[row_idx][col_idx]
            gains = get_gain_matrix(rows, dataset_title, metric, compare_methods, low_snrs)

            for m_idx, method in enumerate(compare_methods):
                style = PUB_STYLE[method]
                offset = (m_idx - (len(compare_methods) - 1) / 2) * width
                bars = ax.bar(
                    x + offset,
                    gains[m_idx],
                    width,
                    color=style['color'],
                    edgecolor='white',
                    linewidth=0.7,
                    hatch=style['hatch'],
                    label=pub_label(method),
                )
                if row_idx == 0 and col_idx == 0:
                    handles.append(bars[0])
                    labels.append(pub_label(method))

            finite = gains[np.isfinite(gains)]
            if finite.size:
                ymin = min(0.0, float(finite.min()))
                ymax = max(0.0, float(finite.max()))
                pad = max((ymax - ymin) * 0.18, 0.035)
                ax.set_ylim(ymin - pad * 0.25, ymax + pad)

            ax.axhline(0, color='#666666', linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{s}' for s in low_snrs])
            if row_idx == n_datasets - 1:
                ax.set_xlabel('SNR (dB)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{dataset_title} - {title}', pad=4)
            ax.grid(True, axis='y', color='#BDBDBD', alpha=0.35, linewidth=0.6)
            ax.set_axisbelow(True)

    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.01), columnspacing=1.6, handlelength=1.9)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.96,
                        hspace=0.62, wspace=0.36)
    save_figure(fig, path)
    plt.close(fig)


def plot_absolute_roi(rows, result_specs, low_snrs, path):
    set_publication_style()
    fig, axes = plt.subplots(
        1,
        len(result_specs),
        figsize=(3.2 * len(result_specs), 2.8),
        squeeze=False,
        constrained_layout=False,
    )
    present_methods = {r['method'] for r in rows}
    methods = [
        m for m in ('cnn', 'semantic', 'sira', 'sira_b1_init', 'sira_b2_init', 'sira_b2_no_r')
        if m in present_methods
    ]
    handles = []
    labels = []

    for ax_idx, (ax, (_, dataset_title, _, _)) in enumerate(zip(axes[0], result_specs)):
        for method in methods:
            values = []
            for snr in low_snrs:
                matches = [
                    r for r in rows
                    if r['dataset'] == dataset_title and r['method'] == method and r['snr'] == snr
                ]
                values.append(matches[0]['roi_psnr'] if matches else np.nan)
            style = PUB_STYLE[method]
            line, = ax.plot(
                low_snrs,
                values,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=1.9,
                markersize=5.5,
                markeredgecolor='white',
                markeredgewidth=0.5,
                label=pub_label(method),
            )
            if ax_idx == 0:
                handles.append(line)
                labels.append(pub_label(method))

        ax.set_title(dataset_title)
        ax.set_xlabel('SNR (dB)')
        if ax_idx == 0:
            ax.set_ylabel('ROI PSNR (dB)')
        ax.grid(True, color='#BDBDBD', alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)

    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01), columnspacing=1.5, handlelength=1.8)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.26, top=0.88,
                        wspace=0.24)
    save_figure(fig, path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default=RESULT_DIR)
    parser.add_argument('--channel', default=CHANNEL)
    parser.add_argument('--latent_ch', type=int, default=LATENT_CH)
    parser.add_argument('--snrs', nargs='+', type=int, default=[-2, 0, 2])
    args = parser.parse_args()

    result_specs = available_results(args.result_dir, args.channel, args.latent_ch)
    if not result_specs:
        raise FileNotFoundError(
            f'No result JSONs found in {args.result_dir} for '
            f'channel={args.channel}, c={args.latent_ch}'
        )

    print('Found result files:')
    for _, title, path, roi_key in result_specs:
        print(f'  {title}: {path} ({roi_key})')

    rows = collect_rows(result_specs, args.snrs)
    print_low_snr_table(rows)

    prefix = os.path.join(args.result_dir, f'low_snr_{args.channel}_c{args.latent_ch}')
    csv_path = f'{prefix}.csv'
    md_path = f'{prefix}.md'
    gain_fig_path = f'{prefix}_gains_pub.png'
    roi_fig_path = f'{prefix}_roi_pub.png'

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    plot_gain_figure(rows, result_specs, args.snrs, gain_fig_path)
    plot_absolute_roi(rows, result_specs, args.snrs, roi_fig_path)

    print(f'\nsaved -> {csv_path}')
    print(f'saved -> {md_path}')
    print(f'saved -> {gain_fig_path}')
    print(f'saved -> {roi_fig_path}')


if __name__ == '__main__':
    main()
