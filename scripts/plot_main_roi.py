"""Plot the c=1 AWGN comparison from the archived per-seed ROI measurements."""

import csv
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS = Path(__file__).resolve().parents[1] / 'results'
STEM = 'sira_main_roi_c1'
DATASETS = {
    'Kodak-24': ('Kodak-24', 'Semantic ROI'),
    'COCO val2017': ('COCO val2017', 'Ground-truth object boxes'),
}
METHODS = {
    'cnn_deepjscc_roi_psnr_db': ('DeepJSCC', '#4F5D75', '--', 'o'),
    'sira_roi_psnr_db': ('SIRA-DeepJSCC', '#007B70', '-', 'D'),
}
SNRS = [-2, 0, 2, 5, 10, 15]
SEEDS = [42, 43, 44]


def load_roi_measurements(path):
    # One row pairs the two methods at the same dataset, SNR and seed.
    with path.open(newline='') as stream:
        rows = {
            (row['dataset'], int(row['snr_db']), int(row['seed'])): tuple(
                float(row[column]) for column in METHODS
            )
            for row in csv.DictReader(stream)
        }
    return {
        dataset: [[rows[dataset, snr, seed] for seed in SEEDS] for snr in SNRS]
        for dataset in DATASETS
    }


def main():
    measurements = load_roi_measurements(RESULTS / f'{STEM}.csv')
    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 5.5), sharex=True, sharey='row',
        gridspec_kw={'height_ratios': [1.8, 1]},
    )
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.09, right=0.97, top=0.74, bottom=0.15,
                        hspace=0.16, wspace=0.17)
    fig.text(0.09, 0.95, 'ROI reconstruction at c = 1',
             fontsize=19, weight='bold', color='#182236')
    fig.text(0.09, 0.902, 'AWGN · Hard water-filling · 3 seeds',
             fontsize=10.5, color='#4F5D75')

    for col, (dataset, (title, roi)) in enumerate(DATASETS.items()):
        top, bottom = axes[:, col]
        paired = measurements[dataset]
        top.set_title(f'{title}\n{roi}', loc='left', fontsize=11, linespacing=1.35)

        for i, (label, color, linestyle, marker) in enumerate(METHODS.values()):
            per_snr = [[pair[i] for pair in group] for group in paired]
            averages = [mean(values) for values in per_snr]
            spreads = [pstdev(values) for values in per_snr]
            top.fill_between(SNRS, [m - s for m, s in zip(averages, spreads)],
                             [m + s for m, s in zip(averages, spreads)],
                             color=color, alpha=0.13, linewidth=0)
            top.plot(SNRS, averages, label=label, color=color, linestyle=linestyle,
                     marker=marker, markerfacecolor='white' if i == 0 else color,
                     markersize=5, linewidth=1.65)

        differences = [[sira - cnn for cnn, sira in group] for group in paired]
        gains = [mean(values) for values in differences]
        bottom.axhline(0, color='#596579', linestyle='--', linewidth=0.9)
        bottom.errorbar(SNRS, gains, yerr=[pstdev(values) for values in differences],
                        fmt='D-', color='#007B70', markersize=4, linewidth=1.3,
                        elinewidth=0.9, capsize=2)
        for axis in (top, bottom):
            axis.set_axisbelow(True)
            axis.grid(axis='y', color='#E2E7ED', linewidth=0.7)
            axis.spines[['top', 'right']].set_visible(False)
            axis.spines[['left', 'bottom']].set_color('#B7C2CE')
            axis.tick_params(labelsize=9, colors='#29364A', length=3)
        bottom.set_xlabel('Channel SNR (dB)', fontsize=10)

    axes[0, 0].set_ylabel('ROI PSNR (dB)', fontsize=10)
    axes[0, 0].set_yticks(range(20, 25))
    axes[0, 0].set_ylim(19.6, 24.75)
    axes[1, 0].set_ylabel('ROI PSNR gain (dB)', fontsize=10)
    axes[1, 0].set_xticks(SNRS)
    axes[1, 0].set_xlim(-2.9, 15.9)
    axes[1, 0].set_ylim(-0.08, 0.65)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc='upper right',
               bbox_to_anchor=(0.978, 0.967), fontsize=10)
    fig.text(0.09, 0.048,
             'Mean ± SD across seeds. Bottom panels show paired differences (SIRA-DeepJSCC − DeepJSCC).',
             fontsize=9, color='#4F5D75')

    fig.savefig(RESULTS / f'{STEM}.png', dpi=200, facecolor='white', metadata={
        'Title': 'ROI reconstruction at c = 1',
        'Description': 'Exp3 AWGN hard water-filling, seeds 42–44; source: sira_main_roi_c1.csv',
    })
    plt.close(fig)


if __name__ == '__main__':
    main()
