#!/usr/bin/env python3
# ============================================================
# visualize_power_map.py
# Visualize SIRA adaptive power maps under different SNRs.
#
# Usage:
#   python -m scripts.visualize_power_map --latent_ch 2
#   python -m scripts.visualize_power_map --latent_ch 2 --image data/kodak/kodim01.png
#   python -m scripts.visualize_power_map --latent_ch 2 --dataset coco --index 12
# ============================================================
import argparse
import glob
import os

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from sira.config import (
    CHANNEL, LATENT_CH, CKPT_DIR, RESULT_DIR, CROP_SIZE,
)
from sira.models import DeepJSCC, DEVICE, SIRA_METHODS


PALETTE = {
    'gray': '#6E7781',
    'blue': '#2C91E0',
    'teal': '#3ABF99',
    'gold': '#F0A73A',
}


def set_publication_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.titlesize': 11,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
    })


def ckpt_path(method, channel=CHANNEL, latent_ch=LATENT_CH):
    return os.path.join(CKPT_DIR, f'{method}_{channel}_c{latent_ch}.pt')


def load_sira(method='sira_b1_init', channel=CHANNEL, latent_ch=LATENT_CH):
    path = ckpt_path(method, channel, latent_ch)
    if method == 'sira_b1_init' and not os.path.isfile(path):
        legacy = ckpt_path('sira', channel, latent_ch)
        if os.path.isfile(legacy):
            print(f'Using legacy SIRA-B1-init checkpoint: {legacy}')
            method, path = 'sira', legacy
    if not os.path.isfile(path):
        raise FileNotFoundError(f'SIRA checkpoint not found: {path}')
    net = DeepJSCC(method=method, latent_ch=latent_ch, channel=channel).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


def list_images(dataset, coco_root='data/coco'):
    if dataset == 'kodak':
        patterns = ['data/kodak/*.png', 'data/kodak/*.jpg']
    elif dataset == 'div2k':
        patterns = [
            'data/DIV2K/DIV2K_valid_HR/*.png',
            'data/DIV2K/DIV2K_valid_HR/*.jpg',
            'data/DIV2K/DIV2K_train_HR/*.png',
            'data/DIV2K/DIV2K_train_HR/*.jpg',
        ]
    elif dataset == 'coco':
        patterns = [
            os.path.join(coco_root, 'val2017', '*.jpg'),
            os.path.join(coco_root, 'val2017', '*.png'),
        ]
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(paths)


def choose_image(args):
    if args.image:
        if not os.path.isfile(args.image):
            raise FileNotFoundError(args.image)
        return args.image

    paths = list_images(args.dataset, args.coco_root)
    if not paths:
        raise FileNotFoundError(
            f'No images found for dataset={args.dataset}. '
            'Pass --image /path/to/image.png to choose one manually.'
        )
    idx = max(0, min(args.index, len(paths) - 1))
    return paths[idx]


def load_image_tensor(path, crop_size=CROP_SIZE):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    side = min(w, h, crop_size)
    img = TF.center_crop(img, [side, side])
    if side != crop_size:
        img = TF.resize(img, [crop_size, crop_size], antialias=True)
    x = TF.to_tensor(img).unsqueeze(0)
    return x, img


def tensor_to_image(x):
    arr = x.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return arr


def normalize_shared(maps):
    stacked = torch.cat([m.reshape(-1) for m in maps])
    lo = stacked.min()
    hi = stacked.max()
    return [(m - lo) / (hi - lo + 1e-8) for m in maps]


@torch.no_grad()
def run_sira(net, x, snrs):
    x = x.to(DEVICE)
    recons = []
    power_maps = []
    raw_power_maps = []

    for snr in snrs:
        x_hat = net(x, float(snr)).clamp(0, 1)
        if not hasattr(net, '_last_sira') or 'power_map' not in net._last_sira:
            raise RuntimeError('SIRA power_map was not produced. Is method="sira"?')
        pm = net._last_sira['power_map'].detach().float()
        pm_up = F.interpolate(pm, size=x.shape[-2:], mode='bilinear', align_corners=False)
        recons.append(x_hat.cpu())
        raw_power_maps.append(pm_up.cpu())

    power_maps = normalize_shared(raw_power_maps)
    return recons, power_maps, raw_power_maps


def overlay_heatmap(image, heat, alpha=0.48, cmap_name='magma'):
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(heat)[..., :3]
    return (1.0 - alpha) * image + alpha * heat_rgb


def add_image_axis(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig, png_path):
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.04)
    pdf_path = os.path.splitext(png_path)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.04)
    print(f'saved -> {png_path}')
    print(f'saved -> {pdf_path}')


def plot_power_maps(x, recons, power_maps, raw_power_maps, snrs, image_path, out_path):
    set_publication_style()
    original = tensor_to_image(x)
    n = len(snrs)
    fig, axes = plt.subplots(
        3,
        n + 1,
        figsize=(2.35 * (n + 1), 6.0),
        constrained_layout=False,
    )

    add_image_axis(axes[0, 0], original, 'Input')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    for j, (snr, x_hat, pm_norm, pm_raw) in enumerate(
        zip(snrs, recons, power_maps, raw_power_maps), start=1
    ):
        recon_img = tensor_to_image(x_hat)
        pm = pm_norm.squeeze().numpy()
        raw = pm_raw.squeeze()
        err = np.mean((recon_img - original) ** 2, axis=-1)
        err = err / (err.max() + 1e-8)

        overlay = overlay_heatmap(original, pm, alpha=0.62, cmap_name='magma')
        err_overlay = overlay_heatmap(recon_img, err, alpha=0.42, cmap_name='viridis')

        add_image_axis(axes[0, j], recon_img, f'Recon @ {snr:g} dB')
        add_image_axis(axes[1, j], overlay, f'Power map @ {snr:g} dB')
        add_image_axis(axes[2, j], err_overlay, f'Error map @ {snr:g} dB')

        # Small concentration cue: coefficient of variation of the power map.
        mean = raw.mean().item()
        std = raw.std().item()
        cv = std / (mean + 1e-8)
        axes[1, j].text(
            0.03, 0.95, f'CV={cv:.2f}',
            transform=axes[1, j].transAxes,
            ha='left', va='top',
            fontsize=8,
            color='white',
            bbox={'facecolor': 'black', 'alpha': 0.35, 'pad': 2, 'edgecolor': 'none'},
        )

    row_labels = ['Reconstruction', 'Adaptive power', 'Error']
    row_y = [0.775, 0.475, 0.175]
    for y, label in zip(row_y, row_labels):
        fig.text(0.018, y, label, ha='left', va='center',
                 rotation=90, fontsize=10, color='#333333')

    basename = os.path.basename(image_path)
    fig.suptitle(f'SIRA Power Allocation Across SNRs ({basename})', y=0.985)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.02, top=0.93,
                        wspace=0.06, hspace=0.12)
    save_figure(fig, out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None,
                        help='optional image path; overrides --dataset/--index')
    parser.add_argument('--dataset', default='kodak',
                        choices=['kodak', 'div2k', 'coco'])
    parser.add_argument('--coco_root', default='data/coco')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--channel', default=CHANNEL)
    parser.add_argument('--latent_ch', type=int, default=LATENT_CH)
    parser.add_argument('--method', default='sira_b1_init',
                        choices=list(SIRA_METHODS),
                        help='SIRA checkpoint variant to visualize')
    parser.add_argument('--snrs', nargs='+', type=float, default=[-2, 5, 15])
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    image_path = choose_image(args)
    out_path = args.out or os.path.join(
        RESULT_DIR, f'power_map_{args.channel}_c{args.latent_ch}.png'
    )

    print(f'Device: {DEVICE}')
    print(f'Image: {image_path}')
    print(f'SNRs: {args.snrs}')

    x, _ = load_image_tensor(image_path, crop_size=CROP_SIZE)
    net = load_sira(method=args.method, channel=args.channel, latent_ch=args.latent_ch)
    recons, power_maps, raw_power_maps = run_sira(net, x, args.snrs)
    plot_power_maps(x, recons, power_maps, raw_power_maps, args.snrs, image_path, out_path)


if __name__ == '__main__':
    main()
