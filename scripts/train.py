#!/usr/bin/env python3
# ============================================================
# train.py  —  训练 B1 / B2 / SIRA，顺序执行
# 用法：python -m scripts.train
#      python -m scripts.train --methods cnn
#      python -m scripts.train --methods sira_b1_init
#      python -m scripts.train --methods sira_b2_init
#      python -m scripts.train --methods sira_b2_no_r
#      python -m scripts.train --channel rayleigh
# ============================================================
import os
import argparse
import random
import json

import torch
import torch.amp
from tqdm import tqdm

from sira import models
from sira.config import (
    CHANNEL, LATENT_CH, EPOCHS_B1, EPOCHS_B2, EPOCHS_SIRA,
    BATCH_SIZE, LR, TRAIN_SNR_RANGE, SNR_SWEEP,
    CKPT_DIR, RESULT_DIR, SEED, IMPORTANCE_MODE,
)
from sira.datasets import get_div2k_loaders
from sira.models import DeepJSCC, loss_fn, DEVICE, SIRA_METHODS


SIRA_INIT_SOURCE = {
    'sira': 'cnn',             # legacy alias: same as SIRA-B1-init
    'sira_b1_init': 'cnn',
    'sira_b2_init': 'semantic',
    'sira_b2_no_r': 'semantic',
}


def make_grad_scaler():
    enabled = DEVICE.type == 'cuda'
    if hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context():
    enabled = DEVICE.type == 'cuda'
    if hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda', enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def seed_everything(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark   = True   # DIV2K 固定尺寸，开 benchmark 更快
    torch.backends.cudnn.deterministic = False


def ckpt_path(method, channel=CHANNEL, latent_ch=LATENT_CH):
    return os.path.join(CKPT_DIR, f'{method}_{channel}_c{latent_ch}.pt')


def load_init_into_sira(net, init_method, channel=CHANNEL, latent_ch=LATENT_CH):
    src = ckpt_path(init_method, channel, latent_ch)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f'{net.method} 需要先训练 {init_method}，checkpoint 不存在：{src}'
        )
    state = torch.load(src, map_location=DEVICE)
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f'loaded {init_method} checkpoint into {net.method} → {src}')
    print(f'  SIRA-only params (missing): {len(missing)} | unexpected: {len(unexpected)}')


def train_one_method(method, channel=CHANNEL, latent_ch=LATENT_CH,
                     epochs=50, batch_size=BATCH_SIZE, lr=LR):

    train_loader, val_loader = get_div2k_loaders(batch_size=batch_size)

    net = DeepJSCC(method=method, latent_ch=latent_ch, channel=channel).to(DEVICE)

    if method in SIRA_METHODS:
        load_init_into_sira(net, SIRA_INIT_SOURCE[method], channel, latent_ch)

    if method in ('semantic',) + SIRA_METHODS and models.IMPORTANCE_MODE == 'dino':
        print(f'[{method}] preloading DINOv2 importance model before training...', flush=True)
        models.get_dinov2_model()

    # Model/DINO initialization may consume RNG state. Reset here so methods
    # receive matched data order, sampled SNRs, and channel-noise sequences.
    seed_everything()

    trainable = [p for p in net.parameters() if p.requires_grad]
    print(f'\n[{method}] trainable params: '
          f'{sum(p.numel() for p in trainable):,} / '
          f'{sum(p.numel() for p in net.parameters()):,}')

    opt   = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    scaler = make_grad_scaler()

    lo, hi = TRAIN_SNR_RANGE
    best_val_psnr = 0.0

    for ep in range(1, epochs + 1):
        # ── train ──
        net.train()
        if method in SIRA_METHODS:
            net.encoder.eval()
            net.decoder.eval()

        running = seen = 0
        pbar = tqdm(train_loader, desc=f'[{method}] ep {ep}/{epochs}', leave=False)
        for x, _ in pbar:
            x   = x.to(DEVICE, non_blocking=True)
            snr = torch.empty(x.shape[0], device=DEVICE).uniform_(lo, hi)
            opt.zero_grad(set_to_none=True)
            with autocast_context():
                x_hat = net(x, snr)
                loss  = loss_fn(net, x, x_hat)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.shape[0]
            seen    += x.shape[0]
            pbar.set_postfix(loss=f'{loss.item():.5f}')
        sched.step()

        # ── val（每 10 epoch 跑一次，省时间）──
        if ep % 10 == 0 or ep == epochs:
            net.eval()
            p_sum = n = 0
            with torch.no_grad():
                for x, _ in val_loader:
                    x     = x.to(DEVICE, non_blocking=True)
                    x_hat = net(x, 5.0).clamp(0, 1)     # 固定 SNR=5 快速估算
                    mse   = torch.nn.functional.mse_loss(x_hat, x).item()
                    import math
                    p_sum += 10 * math.log10(1.0 / (mse + 1e-12))
                    n     += 1
            val_psnr = p_sum / max(n, 1)
            print(f'[{method}|{channel}|c{latent_ch}] '
                  f'ep {ep:3d}/{epochs}  '
                  f'train_loss={running/seen:.5f}  '
                  f'val_PSNR@5dB={val_psnr:.2f}')
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                torch.save(net.state_dict(), ckpt_path(method, channel, latent_ch))
                print(f'  ✓ checkpoint saved (best val PSNR={best_val_psnr:.2f})')
        else:
            print(f'[{method}|{channel}|c{latent_ch}] '
                  f'ep {ep:3d}/{epochs}  loss={running/seen:.5f}')

    # Keep the best validation checkpoint for evaluation; save the last epoch separately.
    final_path = ckpt_path(method, channel, latent_ch).replace('.pt', '_final.pt')
    torch.save(net.state_dict(), final_path)
    print(f'[{method}] best checkpoint → {ckpt_path(method, channel, latent_ch)}')
    print(f'[{method}] final-epoch checkpoint → {final_path}')
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods',  nargs='+',
                        default=['cnn', 'semantic', 'sira_b1_init', 'sira_b2_init'],
                        choices=['cnn', 'semantic'] + list(SIRA_METHODS))
    parser.add_argument('--channel',  default=CHANNEL)
    parser.add_argument('--latent_ch', type=int, default=LATENT_CH)
    parser.add_argument('--epochs_b1', type=int, default=EPOCHS_B1)
    parser.add_argument('--epochs_b2', type=int, default=EPOCHS_B2)
    parser.add_argument('--epochs_sira', type=int, default=EPOCHS_SIRA)
    parser.add_argument('--batch_size',  type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--importance_mode', default=IMPORTANCE_MODE,
                        choices=['edge', 'dino'],
                        help='semantic importance backend; edge avoids DINOv2 download')
    parser.add_argument('--force', action='store_true',
                        help='retrain selected methods even if checkpoints exist')
    args = parser.parse_args()

    models.IMPORTANCE_MODE = args.importance_mode
    seed_everything()
    print(f'Device: {DEVICE}')
    print(f'Methods to train: {args.methods}')
    print(f'Channel: {args.channel}  latent_ch: {args.latent_ch}')
    print(f'Importance mode: {models.IMPORTANCE_MODE}')

    epoch_map = {
        'cnn':      args.epochs_b1,
        'semantic': args.epochs_b2,
        'sira':     args.epochs_sira,
        'sira_b1_init': args.epochs_sira,
        'sira_b2_init': args.epochs_sira,
        'sira_b2_no_r': args.epochs_sira,
    }

    for method in args.methods:
        # Keep data order and newly initialized SIRA modules comparable.
        seed_everything()
        # 如果 checkpoint 已存在就跳过
        path = ckpt_path(method, args.channel, args.latent_ch)
        if os.path.exists(path) and not args.force:
            print(f'\n[{method}] checkpoint already exists, skipping → {path}')
            continue
        train_one_method(
            method=method,
            channel=args.channel,
            latent_ch=args.latent_ch,
            epochs=epoch_map[method],
            batch_size=args.batch_size,
            lr=args.lr,
        )

    print('\n✓ All training done.')


if __name__ == '__main__':
    main()
