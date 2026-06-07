#!/usr/bin/env python3
# ============================================================
# eval.py  —  完整评估：PSNR / SSIM / LPIPS / ROI-PSNR
#             测试集：DIV2K val（域内）+ Kodak-24（域外泛化）
# 用法：python -m scripts.eval
#      python -m scripts.eval --channel rayleigh
#      python -m scripts.eval --skip_kodak
# ============================================================
import os
import argparse
import json
import math

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lpips as lpips_lib
from tqdm import tqdm

from sira import models
from sira.config import (
    CHANNEL, LATENT_CH, SNR_SWEEP, CKPT_DIR, RESULT_DIR,
    METHOD_NAMES, METHOD_STYLE, CROP_SIZE, IMPORTANCE_MODE, SEED,
)
from sira.datasets import get_div2k_loaders, get_kodak_loader
from sira.models import DeepJSCC, semantic_importance, DEVICE, SIRA_METHODS


# ── 指标函数 ──────────────────────────────────────────────────

def batch_psnr(x_hat, x):
    mse = F.mse_loss(x_hat, x, reduction='none').mean(dim=(1,2,3))
    mse = torch.clamp(mse, min=1e-12)
    return (10.0 * torch.log10(1.0 / mse)).mean().item()


def gaussian_window(size=11, sigma=1.5, ch=3, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    w = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return w.expand(ch, 1, size, size).contiguous()


def batch_ssim(x_hat, x, window_size=11, sigma=1.5):
    c   = x.shape[1]
    win = gaussian_window(window_size, sigma, c, x.device)
    pad = window_size // 2
    mu_x  = F.conv2d(x,     win, padding=pad, groups=c)
    mu_y  = F.conv2d(x_hat, win, padding=pad, groups=c)
    mu_x2, mu_y2 = mu_x**2, mu_y**2
    mu_xy = mu_x * mu_y
    sx2 = F.conv2d(x*x,         win, padding=pad, groups=c) - mu_x2
    sy2 = F.conv2d(x_hat*x_hat, win, padding=pad, groups=c) - mu_y2
    sxy = F.conv2d(x*x_hat,     win, padding=pad, groups=c) - mu_xy
    c1, c2 = 0.0001, 0.0009
    ssim_map = ((2*mu_xy+c1)*(2*sxy+c2)) / ((mu_x2+mu_y2+c1)*(sx2+sy2+c2))
    return ssim_map.mean().item()


def ckpt_path(method, channel=CHANNEL, latent_ch=LATENT_CH):
    return os.path.join(CKPT_DIR, f'{method}_{channel}_c{latent_ch}.pt')


def seed_evaluation(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_net(method, channel=CHANNEL, latent_ch=LATENT_CH, input_size=CROP_SIZE):
    path = ckpt_path(method, channel, latent_ch)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    net = DeepJSCC(method=method, latent_ch=latent_ch,
                   channel=channel, input_size=input_size).to(DEVICE)
    net.load_state_dict(torch.load(
        path, map_location=DEVICE
    ))
    net.eval()
    return net


# ── 核心评估函数 ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(net, loader, snr_sweep, lpips_fn, tag=''):
    out = {'snr': list(snr_sweep),
           'psnr': [], 'ssim': [], 'lpips': [],
           'roi_psnr': [], 'bg_psnr': []}

    for snr in snr_sweep:
        p_sum = s_sum = l_sum = 0.0
        roi_sum = bg_sum = 0.0
        n = 0

        for x, _ in tqdm(loader, desc=f'{tag} SNR={snr}dB', leave=False):
            x     = x.to(DEVICE, non_blocking=True)
            x_hat = net(x, float(snr)).clamp(0, 1)
            bs    = x.shape[0]

            p_sum += batch_psnr(x_hat, x) * bs
            s_sum += batch_ssim(x_hat, x) * bs

            # LPIPS：缩放到 256×256
            sz = min(x.shape[-2], x.shape[-1], 256)
            xu  = F.interpolate(x,     size=(sz,sz), mode='bilinear', align_corners=False)
            xhu = F.interpolate(x_hat, size=(sz,sz), mode='bilinear', align_corners=False)
            l_sum += lpips_fn(xu*2-1, xhu*2-1).sum().item()

            # ROI-PSNR: compute per image, then average over images.
            w = semantic_importance(x)
            threshold = w.mean(dim=(1, 2, 3), keepdim=True)
            mask = (w > threshold).float()
            bg = 1.0 - mask
            m3, b3 = mask.expand_as(x), bg.expand_as(x)
            err = (x_hat - x) ** 2
            mse_roi = (err * m3).sum(dim=(1, 2, 3)) / (
                m3.sum(dim=(1, 2, 3)) + 1e-8
            )
            mse_bg = (err * b3).sum(dim=(1, 2, 3)) / (
                b3.sum(dim=(1, 2, 3)) + 1e-8
            )
            roi_sum += (
                10.0 * torch.log10(1.0 / torch.clamp(mse_roi, min=1e-12))
            ).sum().item()
            bg_sum += (
                10.0 * torch.log10(1.0 / torch.clamp(mse_bg, min=1e-12))
            ).sum().item()

            n += bs

        out['psnr'].append(round(p_sum/n, 3))
        out['ssim'].append(round(s_sum/n, 4))
        out['lpips'].append(round(l_sum/n, 4))
        out['roi_psnr'].append(round(roi_sum/n, 3))
        out['bg_psnr'].append(round(bg_sum/n,  3))
        print(f'  {tag} SNR={snr:>4}dB | '
              f'PSNR={p_sum/n:.2f} SSIM={s_sum/n:.4f} LPIPS={l_sum/n:.4f} | '
              f'ROI-PSNR={roi_sum/n:.2f}  BG-PSNR={bg_sum/n:.2f}')
    return out


# ── 画图 ──────────────────────────────────────────────────────

def plot_results(results_dict, snr_sweep, title_suffix, save_prefix):
    metrics = [
        ('psnr',     'PSNR (dB)',  '↑'),
        ('ssim',     'SSIM',       '↑'),
        ('lpips',    'LPIPS',      '↓'),
        ('roi_psnr', 'ROI PSNR (dB)', '↑'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, ylabel, arrow) in zip(axes.flatten(), metrics):
        for m, data in results_dict.items():
            c, mk, ls = METHOD_STYLE.get(m, ('k','x','-'))
            ax.plot(snr_sweep, data[key],
                    color=c, marker=mk, linestyle=ls,
                    linewidth=1.8, markersize=6,
                    label=METHOD_NAMES.get(m, m))
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(f'{ylabel} {arrow}')
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(title_suffix)
    fig.tight_layout()
    fp = os.path.join(RESULT_DIR, f'{save_prefix}.png')
    fig.savefig(fp, dpi=200, bbox_inches='tight')
    print(f'saved → {fp}')
    plt.close(fig)


def print_table(results_dict, snr_sweep, methods):
    header = f"{'SNR':>5}" + "".join(
        f"  {METHOD_NAMES.get(m,m):>28}" for m in methods
    )
    print(header)
    print('-' * len(header))
    for i, snr in enumerate(snr_sweep):
        row = f'{snr:>5}'
        for m in methods:
            d = results_dict[m]
            row += f"  PSNR={d['psnr'][i]:5.2f} ROI={d['roi_psnr'][i]:5.2f} SSIM={d['ssim'][i]:.4f}"
        print(row)

    # ΔROI vs B1
    sira_methods = [m for m in SIRA_METHODS if m in results_dict]
    if sira_methods and 'cnn' in results_dict:
        for method in sira_methods:
            name = METHOD_NAMES.get(method, method)
            print(f"\n{name}:")
            print(f"{'SNR':>5} | {'ROI-PSNR':>9} | {'BG-PSNR':>8} | {'ΔROI vs B1':>11}")
            print('-' * 43)
            for i, snr in enumerate(snr_sweep):
                roi  = results_dict[method]['roi_psnr'][i]
                bg   = results_dict[method]['bg_psnr'][i]
                b1   = results_dict['cnn']['roi_psnr'][i]
                print(f'{snr:>5} | {roi:>9.2f} | {bg:>8.2f} | {roi-b1:>+11.2f}')


# ── main ──────────────────────────────────────────────────────

def main():
    global RESULT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods',    nargs='+',
                        default=['cnn','semantic','sira_b1_init','sira_b2_init'],
                        choices=['cnn', 'semantic'] + list(SIRA_METHODS))
    parser.add_argument('--channel',    default=CHANNEL)
    parser.add_argument('--latent_ch',  type=int, default=LATENT_CH)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--result_dir', default=RESULT_DIR)
    parser.add_argument('--seed', type=int, default=SEED,
                        help='reset before each method for matched channel noise')
    parser.add_argument('--skip_div2k', action='store_true')
    parser.add_argument('--skip_kodak', action='store_true')
    parser.add_argument('--importance_mode', default=IMPORTANCE_MODE,
                        choices=['edge', 'dino'],
                        help='semantic importance backend used for ROI metrics')
    args = parser.parse_args()

    RESULT_DIR = args.result_dir
    os.makedirs(RESULT_DIR, exist_ok=True)
    models.IMPORTANCE_MODE = args.importance_mode
    print(f'Device: {DEVICE}')
    print(f'Importance mode: {models.IMPORTANCE_MODE}')
    lpips_fn = lpips_lib.LPIPS(net='alex').to(DEVICE)
    lpips_fn.eval()
    if models.IMPORTANCE_MODE == 'dino':
        print('Preloading DINOv2 before matched-noise evaluation...')
        models.get_dinov2_model()

    all_results = {}

    # ── DIV2K val ──
    if not args.skip_div2k:
        print('\n' + '='*70)
        print('DIV2K Validation Set')
        print('='*70)
        _, val_loader = get_div2k_loaders(batch_size=args.batch_size)
        div2k_results = {}
        for m in args.methods:
            print(f'\n── {METHOD_NAMES.get(m,m)} ──')
            net = load_net(m, args.channel, args.latent_ch, input_size=CROP_SIZE)
            seed_evaluation(args.seed)
            div2k_results[m] = evaluate_dataset(
                net, val_loader, SNR_SWEEP, lpips_fn, tag=m
            )
        print('\n── Summary ──')
        print_table(div2k_results, SNR_SWEEP, args.methods)
        plot_results(div2k_results, SNR_SWEEP,
                     f'DIV2K Val  ({args.channel.upper()}, c={args.latent_ch})',
                     f'div2k_{args.channel}_c{args.latent_ch}')
        fp = os.path.join(RESULT_DIR, f'div2k_{args.channel}_c{args.latent_ch}.json')
        with open(fp, 'w') as f:
            json.dump({'dataset': 'DIV2K_val',
                       'evaluation_seed': args.seed,
                       'snr_sweep': SNR_SWEEP,
                       'methods': div2k_results}, f, indent=2)
        print(f'saved → {fp}')
        all_results['div2k'] = div2k_results

    # ── Kodak ──
    if not args.skip_kodak:
        print('\n' + '='*70)
        print('Kodak-24  (zero-shot generalization)')
        print('='*70)
        kodak_loader  = get_kodak_loader(num_workers=2)
        kodak_results = {}
        for m in args.methods:
            print(f'\n── {METHOD_NAMES.get(m,m)} ──')
            # Kodak 是整图推理，input_size 对应 Kodak 的实际尺寸
            # DeepJSCC 是全卷积，传任何 input_size 都能跑；
            # 但 SemanticPriorMapper 的 latent_hw 需要和 encoder 输出一致
            # → 用 input_size=512（Kodak 短边）让 latent_hw 计算正确
            net = load_net(m, args.channel, args.latent_ch, input_size=CROP_SIZE)
            seed_evaluation(args.seed)
            kodak_results[m] = evaluate_dataset(
                net, kodak_loader, SNR_SWEEP, lpips_fn, tag=m
            )
        print('\n── Summary ──')
        print_table(kodak_results, SNR_SWEEP, args.methods)
        plot_results(kodak_results, SNR_SWEEP,
                     f'Kodak-24  ({args.channel.upper()}, c={args.latent_ch})',
                     f'kodak_{args.channel}_c{args.latent_ch}')
        fp = os.path.join(RESULT_DIR, f'kodak_{args.channel}_c{args.latent_ch}.json')
        with open(fp, 'w') as f:
            json.dump({'dataset': 'Kodak-24',
                       'evaluation_seed': args.seed,
                       'snr_sweep': SNR_SWEEP,
                       'methods': kodak_results}, f, indent=2)
        print(f'saved → {fp}')
        all_results['kodak'] = kodak_results

    fp = os.path.join(RESULT_DIR, f'all_{args.channel}_c{args.latent_ch}.json')
    with open(fp, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n✓ All results saved → {fp}')


if __name__ == '__main__':
    main()
