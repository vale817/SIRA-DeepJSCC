#!/usr/bin/env python3
# ============================================================
# eval_coco.py  —  COCO object-ROI evaluation
# 用法：
#   python -m scripts.eval_coco
#   python -m scripts.eval_coco --coco_root data/coco
#   python -m scripts.eval_coco --max_images 500
# ============================================================
import argparse
import json
import math
import os
from collections import defaultdict

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import lpips as lpips_lib
from tqdm import tqdm

from sira.config import (
    CHANNEL, LATENT_CH, SNR_SWEEP, CKPT_DIR, RESULT_DIR,
    METHOD_NAMES, METHOD_STYLE, CROP_SIZE, SEED,
)
from sira.models import DeepJSCC, DEVICE, SIRA_METHODS


def batch_psnr(x_hat, x):
    mse = F.mse_loss(x_hat, x, reduction='none').mean(dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    return (10.0 * torch.log10(1.0 / mse)).mean().item()


def gaussian_window(size=11, sigma=1.5, ch=3, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return w.expand(ch, 1, size, size).contiguous()


def batch_ssim(x_hat, x, window_size=11, sigma=1.5):
    c = x.shape[1]
    win = gaussian_window(window_size, sigma, c, x.device)
    pad = window_size // 2
    mu_x = F.conv2d(x, win, padding=pad, groups=c)
    mu_y = F.conv2d(x_hat, win, padding=pad, groups=c)
    mu_x2, mu_y2 = mu_x ** 2, mu_y ** 2
    mu_xy = mu_x * mu_y
    sx2 = F.conv2d(x * x, win, padding=pad, groups=c) - mu_x2
    sy2 = F.conv2d(x_hat * x_hat, win, padding=pad, groups=c) - mu_y2
    sxy = F.conv2d(x * x_hat, win, padding=pad, groups=c) - mu_xy
    c1, c2 = 0.0001, 0.0009
    ssim_map = ((2 * mu_xy + c1) * (2 * sxy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2)
    )
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
    net = DeepJSCC(
        method=method,
        latent_ch=latent_ch,
        channel=channel,
        input_size=input_size,
    ).to(DEVICE)
    net.load_state_dict(torch.load(
        path, map_location=DEVICE
    ))
    net.eval()
    return net


def find_coco_paths(coco_root):
    image_candidates = [
        os.path.join(coco_root, 'val2017'),
        os.path.join(coco_root, 'images', 'val2017'),
        os.path.join(coco_root, 'coco', 'val2017'),
    ]
    ann_candidates = [
        os.path.join(coco_root, 'annotations', 'instances_val2017.json'),
        os.path.join(coco_root, 'annotations_trainval2017', 'annotations', 'instances_val2017.json'),
        os.path.join(coco_root, 'instances_val2017.json'),
    ]

    image_dir = next((p for p in image_candidates if os.path.isdir(p)), None)
    ann_file = next((p for p in ann_candidates if os.path.isfile(p)), None)

    if image_dir is None or ann_file is None:
        raise FileNotFoundError(
            'COCO val2017 images or annotations were not found.\n'
            f'Checked image dirs: {image_candidates}\n'
            f'Checked annotation files: {ann_candidates}\n\n'
            'If your files are still zipped, run for example:\n'
            '  mkdir -p data/coco\n'
            '  unzip /autodl-pub/COCO2017/val2017.zip -d data/coco\n'
            '  unzip /autodl-pub/COCO2017/annotations_trainval2017.zip -d data/coco\n'
            '  python -m scripts.eval_coco --coco_root data/coco'
        )
    return image_dir, ann_file


class CocoObjectROIDataset(Dataset):
    def __init__(self, coco_root='data/coco',
                 max_images=None, category_names=None):
        self.image_dir, ann_file = find_coco_paths(coco_root)
        with open(ann_file, 'r') as f:
            data = json.load(f)

        cat_name_to_id = {c['name']: c['id'] for c in data.get('categories', [])}
        if category_names:
            missing = sorted(set(category_names) - set(cat_name_to_id))
            if missing:
                raise ValueError(f'Unknown COCO categories: {missing}')
            keep_cat_ids = {cat_name_to_id[name] for name in category_names}
        else:
            keep_cat_ids = None

        anns_by_image = defaultdict(list)
        for ann in data['annotations']:
            if ann.get('iscrowd', 0):
                continue
            if keep_cat_ids is not None and ann['category_id'] not in keep_cat_ids:
                continue
            x, y, w, h = ann['bbox']
            if w <= 1 or h <= 1:
                continue
            anns_by_image[ann['image_id']].append(ann)

        records = []
        for img in data['images']:
            anns = anns_by_image.get(img['id'], [])
            if not anns:
                continue
            fp = os.path.join(self.image_dir, img['file_name'])
            if os.path.isfile(fp):
                records.append((img, anns))

        records = sorted(records, key=lambda item: item[0]['file_name'])
        if max_images is not None:
            records = records[:max_images]

        if not records:
            raise RuntimeError('No COCO validation images with object annotations were found.')

        self.records = records
        print(f'COCO Object ROI dataset: {len(self.records)} images')
        print(f'  images: {self.image_dir}')
        print(f'  anns:   {ann_file}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_info, anns = self.records[idx]
        path = os.path.join(self.image_dir, img_info['file_name'])
        img = Image.open(path).convert('RGB')
        x = TF.to_tensor(img)

        _, h, w = x.shape
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        for ann in anns:
            bx, by, bw, bh = ann['bbox']
            x1 = max(0, int(math.floor(bx)))
            y1 = max(0, int(math.floor(by)))
            x2 = min(w, int(math.ceil(bx + bw)))
            y2 = min(h, int(math.ceil(by + bh)))
            if x2 > x1 and y2 > y1:
                mask[:, y1:y2, x1:x2] = 1.0

        # Keep spatial sizes even so the stride-2 encoder and transposed
        # decoder return exactly the same image size.
        even_h = h - (h % 2)
        even_w = w - (w % 2)
        x = x[:, :even_h, :even_w]
        mask = mask[:, :even_h, :even_w]

        return x, mask, img_info['file_name']


def coco_collate(batch):
    # batch_size should be 1 because COCO images have different sizes.
    x, mask, name = batch[0]
    return x.unsqueeze(0), mask.unsqueeze(0), name


@torch.no_grad()
def evaluate_dataset(net, loader, snr_sweep, lpips_fn, tag=''):
    out = {
        'snr': list(snr_sweep),
        'psnr': [], 'ssim': [], 'lpips': [],
        'object_roi_psnr': [], 'background_psnr': [], 'roi_ratio': [],
    }

    for snr in snr_sweep:
        p_sum = s_sum = l_sum = 0.0
        roi_sum = bg_sum = ratio_sum = 0.0
        n = 0

        for x, mask, _ in tqdm(loader, desc=f'{tag} COCO SNR={snr}dB', leave=False):
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            x_hat = net(x, float(snr)).clamp(0, 1)

            p_sum += batch_psnr(x_hat, x)
            s_sum += batch_ssim(x_hat, x)

            sz = min(x.shape[-2], x.shape[-1], 256)
            xu = F.interpolate(x, size=(sz, sz), mode='bilinear', align_corners=False)
            xhu = F.interpolate(x_hat, size=(sz, sz), mode='bilinear', align_corners=False)
            l_sum += lpips_fn(xu * 2 - 1, xhu * 2 - 1).sum().item()

            m3 = mask.expand_as(x)
            b3 = 1.0 - m3
            err = (x_hat - x) ** 2

            mse_roi = (err * m3).sum() / (m3.sum() + 1e-8)
            mse_bg = (err * b3).sum() / (b3.sum() + 1e-8)
            roi_sum += 10.0 * math.log10(1.0 / (mse_roi.item() + 1e-12))
            bg_sum += 10.0 * math.log10(1.0 / (mse_bg.item() + 1e-12))
            ratio_sum += mask.mean().item()
            n += 1

        out['psnr'].append(round(p_sum / n, 3))
        out['ssim'].append(round(s_sum / n, 4))
        out['lpips'].append(round(l_sum / n, 4))
        out['object_roi_psnr'].append(round(roi_sum / n, 3))
        out['background_psnr'].append(round(bg_sum / n, 3))
        out['roi_ratio'].append(round(ratio_sum / n, 4))
        print(
            f'  {tag} SNR={snr:>4}dB | '
            f'PSNR={p_sum/n:.2f} SSIM={s_sum/n:.4f} LPIPS={l_sum/n:.4f} | '
            f'ObjROI={roi_sum/n:.2f} BG={bg_sum/n:.2f}'
        )

    return out


def plot_results(results_dict, snr_sweep, title_suffix, save_prefix):
    metrics = [
        ('psnr', 'PSNR (dB)', 'up'),
        ('ssim', 'SSIM', 'up'),
        ('lpips', 'LPIPS', 'down'),
        ('object_roi_psnr', 'Object ROI PSNR (dB)', 'up'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, ylabel, arrow) in zip(axes.flatten(), metrics):
        for m, data in results_dict.items():
            c, mk, ls = METHOD_STYLE.get(m, ('k', 'x', '-'))
            ax.plot(
                snr_sweep,
                data[key],
                color=c,
                marker=mk,
                linestyle=ls,
                linewidth=1.8,
                markersize=6,
                label=METHOD_NAMES.get(m, m),
            )
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(f'{ylabel} {arrow}')
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(title_suffix)
    fig.tight_layout()
    fp = os.path.join(RESULT_DIR, f'{save_prefix}.png')
    fig.savefig(fp, dpi=200, bbox_inches='tight')
    print(f'saved -> {fp}')
    plt.close(fig)


def print_table(results_dict, snr_sweep, methods):
    header = f"{'SNR':>5}" + ''.join(
        f"  {METHOD_NAMES.get(m, m):>28}" for m in methods
    )
    print(header)
    print('-' * len(header))
    for i, snr in enumerate(snr_sweep):
        row = f'{snr:>5}'
        for m in methods:
            d = results_dict[m]
            row += (
                f"  PSNR={d['psnr'][i]:5.2f} "
                f"ObjROI={d['object_roi_psnr'][i]:5.2f} "
                f"SSIM={d['ssim'][i]:.4f}"
            )
        print(row)

    if 'cnn' in results_dict:
        for method in [m for m in SIRA_METHODS if m in results_dict]:
            name = METHOD_NAMES.get(method, method)
            print(f'\n{name}:')
            print(f"{'SNR':>5} | {'ObjROI':>9} | {'ObjROI B1':>9} | {'Delta':>8}")
            print('-' * 41)
            for i, snr in enumerate(snr_sweep):
                sira = results_dict[method]['object_roi_psnr'][i]
                b1 = results_dict['cnn']['object_roi_psnr'][i]
                print(f'{snr:>5} | {sira:>9.2f} | {b1:>9.2f} | {sira-b1:>+8.3f}')


def main():
    global RESULT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_root', default='data/coco')
    parser.add_argument('--methods', nargs='+',
                        default=['cnn', 'semantic', 'sira_b1_init', 'sira_b2_init'],
                        choices=['cnn', 'semantic'] + list(SIRA_METHODS))
    parser.add_argument('--channel', default=CHANNEL)
    parser.add_argument('--latent_ch', type=int, default=LATENT_CH)
    parser.add_argument('--max_images', type=int, default=None,
                        help='optional quick-test limit, e.g. 500')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--result_dir', default=RESULT_DIR)
    parser.add_argument('--seed', type=int, default=SEED,
                        help='reset before each method for matched channel noise')
    parser.add_argument('--categories', nargs='*', default=None,
                        help='optional COCO category names, e.g. person car bus')
    args = parser.parse_args()

    RESULT_DIR = args.result_dir
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f'Device: {DEVICE}')
    print(f'COCO root: {args.coco_root}')
    print('ROI source: COCO bounding boxes')

    dataset = CocoObjectROIDataset(
        coco_root=args.coco_root,
        max_images=args.max_images,
        category_names=args.categories,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=coco_collate,
    )

    lpips_fn = lpips_lib.LPIPS(net='alex').to(DEVICE)
    lpips_fn.eval()

    coco_results = {}
    for m in args.methods:
        print(f'\n-- {METHOD_NAMES.get(m, m)} --')
        net = load_net(m, args.channel, args.latent_ch, input_size=CROP_SIZE)
        seed_evaluation(args.seed)
        coco_results[m] = evaluate_dataset(
            net, loader, SNR_SWEEP, lpips_fn, tag=m
        )

    print('\n-- Summary --')
    print_table(coco_results, SNR_SWEEP, args.methods)

    suffix = f'COCO val2017 Object ROI  ({args.channel.upper()}, c={args.latent_ch})'
    prefix = f'coco_object_{args.channel}_c{args.latent_ch}'
    plot_results(coco_results, SNR_SWEEP, suffix, prefix)

    fp = os.path.join(RESULT_DIR, f'{prefix}.json')
    with open(fp, 'w') as f:
        json.dump({
            'dataset': 'COCO_val2017',
            'roi_source': 'bbox',
            'evaluation_seed': args.seed,
            'snr_sweep': SNR_SWEEP,
            'methods': coco_results,
        }, f, indent=2)
    print(f'saved -> {fp}')


if __name__ == '__main__':
    main()
