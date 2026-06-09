# ============================================================
# models.py  —  所有模型定义（与 notebook 完全一致）
# ============================================================
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    IMPORTANCE_MODE, DINO_MODEL_NAME, DINO_INPUT_SIZE, DINO_TEMPERATURE,
    DINO_REC_ALPHA, DINO_M_LAMBDA, DINO_HUB_DIR, DINO_SOURCE,
    DINO_REPO_OR_DIR, CROP_SIZE, ALLOCATION_MODE,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DIV2K 256×256 crop，encoder 做一次 stride-2 下采样 → latent 128×128
# notebook 里 CIFAR-10 是 32×32 → latent 16×16
# SemanticPriorMapper 的 latent_hw 会在 DeepJSCC.__init__ 里自动计算
_DINO_IMPORTANCE_MODEL = None


# ── 基础模块 ──────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(ch)
        self.act   = nn.PReLU()

    def forward(self, x):
        r = self.act(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        return self.act(x + r)


class Encoder(nn.Module):
    def __init__(self, latent_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   64,  5, 2, 2)   # stride-2 下采样
        self.bn1   = nn.BatchNorm2d(64)
        self.act1  = nn.PReLU()
        self.res1  = ResBlock(64)
        self.conv2 = nn.Conv2d(64,  128, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(128)
        self.act2  = nn.PReLU()
        self.res2  = ResBlock(128)
        self.conv3 = nn.Conv2d(128, latent_ch, 3, 1, 1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.res2(x)
        return self.conv3(x)


class Decoder(nn.Module):
    def __init__(self, latent_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_ch, 128, 3, 1, 1)
        self.bn1   = nn.BatchNorm2d(128)
        self.act1  = nn.PReLU()
        self.res1  = ResBlock(128)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(64)
        self.act2  = nn.PReLU()
        self.res2  = ResBlock(64)
        self.up    = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # stride-2 上采样
        self.bn3   = nn.BatchNorm2d(64)
        self.act3  = nn.PReLU()
        self.out   = nn.Conv2d(64, 3, 5, 1, 2)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.res2(x)
        x = self.act3(self.bn3(self.up(x)))
        return torch.sigmoid(self.out(x))


# ── 信道 ──────────────────────────────────────────────────────

def power_normalize(z):
    b  = z.shape[0]
    zf = z.reshape(b, -1)
    n  = zf.shape[1]
    norm = torch.sqrt((zf ** 2).sum(dim=1, keepdim=True) + 1e-9)
    zf = zf * (n ** 0.5) / norm
    return zf.view_as(z)


class Channel(nn.Module):
    def __init__(self, kind='awgn'):
        super().__init__()
        assert kind in ('awgn', 'rayleigh')
        self.kind = kind

    def forward(self, z, snr_db):
        snr_lin = 10.0 ** (snr_db / 10.0)
        sigma   = torch.sqrt(1.0 / (2.0 * snr_lin))
        if self.kind == 'awgn':
            return z + sigma * torch.randn_like(z)
        h = torch.sqrt(
            torch.randn(z.shape[0], 1, 1, 1, device=z.device) ** 2 +
            torch.randn(z.shape[0], 1, 1, 1, device=z.device) ** 2
        ) / (2 ** 0.5)
        y = h * z + sigma * torch.randn_like(z)
        return y / (h + 1e-9)


# ── 重要性估计 ────────────────────────────────────────────────

def edge_importance(x):
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    mag = torch.sqrt(
        F.conv2d(gray, kx, padding=1) ** 2 +
        F.conv2d(gray, ky, padding=1) ** 2 + 1e-12
    )
    return mag / (mag.mean(dim=(1,2,3), keepdim=True) + 1e-8)


class DINOv2Importance(nn.Module):
    def __init__(self, model_name=DINO_MODEL_NAME,
                 input_size=DINO_INPUT_SIZE, temperature=DINO_TEMPERATURE):
        super().__init__()
        self.input_size  = input_size
        self.temperature = temperature
        if DINO_HUB_DIR:
            os.makedirs(DINO_HUB_DIR, exist_ok=True)
            torch.hub.set_dir(DINO_HUB_DIR)

        hub_dir = torch.hub.get_dir()
        print(
            f'DINOv2 hub load: repo={DINO_REPO_OR_DIR} '
            f'model={model_name} source={DINO_SOURCE} cache={hub_dir}',
            flush=True,
        )
        print(
            'If this is the first run, torch.hub may download the DINOv2 '
            'repo and checkpoint before the first training batch starts.',
            flush=True,
        )
        try:
            self.backbone = torch.hub.load(
                DINO_REPO_OR_DIR,
                model_name,
                source=DINO_SOURCE,
                trust_repo=True,
            )
        except Exception as exc:
            raise RuntimeError(
                'Failed to load DINOv2 via torch.hub. If AutoDL cannot access '
                'GitHub/checkpoint URLs, either run with '
                '`SIRA_IMPORTANCE_MODE=edge python -m scripts.train ...`, or download '
                'facebookresearch/dinov2 to the machine and run with '
                '`SIRA_DINO_SOURCE=local '
                'SIRA_DINO_REPO_OR_DIR=/path/to/dinov2 python -m scripts.train ...`.'
            ) from exc
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer('mean', mean)
        self.register_buffer('std',  std)

    @torch.no_grad()
    def forward(self, x, out_size=None):
        out_size = out_size or x.shape[-2:]
        xd = F.interpolate(x.float(), size=(self.input_size, self.input_size),
                           mode='bilinear', align_corners=False)
        xd = (xd - self.mean) / self.std
        feats = self.backbone.forward_features(xd)
        patch = feats['x_norm_patchtokens']
        cls   = feats['x_norm_clstoken']
        sim   = F.cosine_similarity(patch, cls.unsqueeze(1), dim=-1)
        score = F.softplus(sim / self.temperature) + 1e-6
        b, n  = score.shape
        h = w = int(n ** 0.5)
        score = score.view(b, 1, h, w)
        score = F.interpolate(score, size=out_size, mode='bilinear', align_corners=False)
        return score / (score.mean(dim=(1,2,3), keepdim=True) + 1e-8)


def get_dinov2_model():
    global _DINO_IMPORTANCE_MODEL
    if _DINO_IMPORTANCE_MODEL is None:
        print(f'Loading DINOv2: {DINO_MODEL_NAME}', flush=True)
        _DINO_IMPORTANCE_MODEL = DINOv2Importance().to(DEVICE)
        _DINO_IMPORTANCE_MODEL.eval()
        print('DINOv2 loaded.', flush=True)
    return _DINO_IMPORTANCE_MODEL


@torch.no_grad()
def dinov2_importance(x):
    return get_dinov2_model()(x, out_size=x.shape[-2:])


def semantic_importance(x):
    if IMPORTANCE_MODE == 'edge':
        return edge_importance(x)
    if IMPORTANCE_MODE == 'dino':
        return dinov2_importance(x)
    raise ValueError(f'Unknown IMPORTANCE_MODE: {IMPORTANCE_MODE}')


# ── SIRA 三模块 ───────────────────────────────────────────────

def mean_normalize(x, dims, eps=1e-8):
    return x / (x.mean(dim=dims, keepdim=True) + eps)


def hard_power_project(p, eps=1e-8):
    b = p.shape[0]
    n = p[0].numel()
    denom = p.reshape(b, -1).sum(dim=1).view(b, *([1] * (p.dim() - 1)))
    return n * p / (denom + eps)


def expand_to_latent_symbols(x, z):
    b, c, h, w = z.shape
    if x.shape[-2:] != (h, w):
        x = F.interpolate(x, size=(h, w), mode='area')
    if x.shape[1] == 1:
        x = x.expand(b, c, h, w)
    elif x.shape[1] != c:
        x = x.mean(dim=1, keepdim=True).expand(b, c, h, w)
    return x


def water_filling_power(risk, gamma=None, eps=1e-8, iters=32):
    """Closed-form SIRA prior: P_i=(sqrt(s_i/gamma)/mu - 1/gamma)^+."""
    b = risk.shape[0]
    flat = risk.clamp_min(eps).reshape(b, -1)
    n = flat.shape[1]

    if gamma is None:
        return hard_power_project(torch.sqrt(risk.clamp_min(eps)), eps=eps)

    gamma = gamma.view(b, 1).float().clamp_min(eps)
    inv_gamma = 1.0 / gamma
    gain = torch.sqrt(flat / gamma)
    lo = torch.full((b, 1), eps, dtype=flat.dtype, device=flat.device)
    hi = torch.sqrt(flat * gamma).max(dim=1, keepdim=True).values.clamp_min(eps)

    for _ in range(iters):
        mu = 0.5 * (lo + hi)
        p = torch.relu(gain / mu - inv_gamma)
        too_much = p.sum(dim=1, keepdim=True) > n
        lo = torch.where(too_much, mu, lo)
        hi = torch.where(too_much, hi, mu)

    p = torch.relu(gain / hi.clamp_min(eps) - inv_gamma)
    p = p.view_as(risk)
    return hard_power_project(p + eps, eps=eps)


def stabilized_water_filling_power(risk, gamma=None, pivot_snr_db=2.0, eps=1e-8):
    """Blend hard water-filling with sqrt-risk allocation at low SNR."""
    p_sqrt_risk = hard_power_project(torch.sqrt(risk.clamp_min(eps)), eps=eps)
    p_waterfill = water_filling_power(risk, gamma=gamma, eps=eps)
    b = risk.shape[0]

    if gamma is None:
        blend_lambda = torch.ones(b, 1, 1, 1, device=risk.device, dtype=risk.dtype)
    else:
        gamma = gamma.view(b, 1, 1, 1).float().clamp_min(eps)
        pivot_gamma = 10.0 ** (pivot_snr_db / 10.0)
        blend_lambda = (pivot_gamma / (gamma + pivot_gamma)).to(risk.dtype)

    p_stabilized = hard_power_project(
        (1.0 - blend_lambda) * p_waterfill + blend_lambda * p_sqrt_risk,
        eps=eps,
    )
    return p_stabilized, p_waterfill, p_sqrt_risk, blend_lambda


def effective_power_from_transmit(z_alloc, power_symbol, eps=1e-8):
    b = z_alloc.shape[0]
    n = z_alloc[0].numel()
    energy = z_alloc.reshape(b, -1).pow(2).sum(dim=1).view(b, 1, 1, 1)
    alpha = n / (energy + eps)
    return alpha * power_symbol, alpha


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch), nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch), nn.PReLU(),
        )
    def forward(self, x):
        return self.net(x)


class SemanticPriorMapper(nn.Module):
    """M: image → importance map（下采样到 latent 分辨率）"""
    def __init__(self, latent_hw, base_ch=32):
        super().__init__()
        self.latent_hw = latent_hw
        self.stem  = ConvBlock(3, base_ch)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch,     base_ch*2, 3, 2, 1),
            nn.BatchNorm2d(base_ch*2), nn.PReLU(),
            ConvBlock(base_ch*2, base_ch*2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, 2, 1),
            nn.BatchNorm2d(base_ch*4), nn.PReLU(),
            ConvBlock(base_ch*4, base_ch*4),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(base_ch + base_ch*2 + base_ch*4, base_ch*2, 1),
            nn.PReLU(),
            nn.Conv2d(base_ch*2, 1, 1),
        )

    def forward(self, x):
        h0 = self.stem(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        h1 = F.interpolate(h1, size=h0.shape[-2:], mode='bilinear', align_corners=False)
        h2 = F.interpolate(h2, size=h0.shape[-2:], mode='bilinear', align_corners=False)
        m_pix = F.softplus(self.fuse(torch.cat([h0, h1, h2], dim=1))) + 1e-6
        m = F.interpolate(m_pix, size=self.latent_hw, mode='area')
        m = mean_normalize(m, dims=(1,2,3))
        return m, m_pix


class ChannelReliabilityMapper(nn.Module):
    """R: SNR dB -> embedding, linear gamma, and concentration temperature."""
    def __init__(self, embed_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.PReLU(),
            nn.Linear(32, embed_dim), nn.PReLU(),
        )
        self.temp_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softplus(),
        )

    def forward(self, snr):
        snr = snr.view(snr.shape[0], 1).float()
        gamma = torch.pow(10.0, snr / 10.0).squeeze(-1)
        embed = self.net(snr / 20.0)
        tau = self.temp_head(embed).squeeze(-1) + 0.5
        return embed, gamma, tau


class ProtectionAdapter(nn.Module):
    """A: semantic risk + channel state -> constrained symbol-level power."""
    def __init__(self, latent_ch, r_dim=8, hidden=32,
                 allocation_mode=ALLOCATION_MODE):
        super().__init__()
        if allocation_mode not in ('hard', 'soft'):
            raise ValueError(f'Unknown allocation_mode: {allocation_mode}')
        self.latent_ch = latent_ch
        self.allocation_mode = allocation_mode
        self.net = nn.Sequential(
            nn.Conv2d(2 * latent_ch + r_dim, hidden, 3, 1, 1), nn.PReLU(),
            nn.Conv2d(hidden,    hidden, 3, 1, 1), nn.PReLU(),
            nn.Conv2d(hidden,    latent_ch, 1),
        )
        self.residual_mix_logit = nn.Parameter(torch.tensor(-3.0))

    def forward(self, z, semantic_importance, vulnerability=None,
                r_embed=None, gamma=None, tau=None):
        b, c, h, w = z.shape
        m = expand_to_latent_symbols(semantic_importance, z)
        m = mean_normalize(m.clamp_min(1e-8), dims=(1,2,3))

        if vulnerability is None:
            v = torch.ones_like(m)
        else:
            v = expand_to_latent_symbols(vulnerability.detach(), z)
            v = mean_normalize(v.clamp_min(1e-8), dims=(1,2,3))

        risk = mean_normalize((m * v).clamp_min(1e-8), dims=(1,2,3))
        p_soft, p_waterfill, p_sqrt_risk, soft_blend_lambda = (
            stabilized_water_filling_power(risk, gamma=gamma)
        )
        if self.allocation_mode == 'soft':
            p_prior = p_soft
            blend_lambda = soft_blend_lambda
        else:
            p_prior = p_waterfill
            blend_lambda = torch.zeros_like(soft_blend_lambda)

        adapter_risk = risk
        if tau is not None:
            tau_ = tau.view(b, 1, 1, 1).float().clamp(0.25, 4.0)
            adapter_risk = mean_normalize(
                risk.float().clamp_min(1e-8) ** tau_,
                dims=(1,2,3),
            ).to(risk.dtype)

        adapter_input = torch.cat([adapter_risk, p_prior.detach()], dim=1)
        if r_embed is not None:
            r = r_embed.view(b, -1, 1, 1).expand(b, -1, h, w)
            adapter_input = torch.cat([adapter_input, r], dim=1)

        p_residual = hard_power_project(F.softplus(self.net(adapter_input)) + 1e-6)
        mix = torch.sigmoid(self.residual_mix_logit)
        p_symbol = hard_power_project((1.0 - mix) * p_prior + mix * p_residual)
        z_out = torch.sqrt(p_symbol + 1e-8) * z
        allocation = {
            'power_prior': p_prior,
            'power_waterfill': p_waterfill,
            'power_sqrt_risk': p_sqrt_risk,
            'low_snr_blend_lambda': blend_lambda,
            'residual_mix': mix,
            'allocation_mode': self.allocation_mode,
        }
        return z_out, p_symbol, risk, allocation


# ── DeepJSCC 主模型 ───────────────────────────────────────────

SIRA_NO_R_METHODS = ('sira_b2_no_r',)
SIRA_METHODS = ('sira', 'sira_b1_init', 'sira_b2_init') + SIRA_NO_R_METHODS
SEMANTIC_LOSS_METHODS = ('semantic',) + SIRA_METHODS


def load_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible = {
        k: v for k, v in state_dict.items()
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
    }
    skipped = sorted(k for k in state_dict if k not in compatible)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return missing, unexpected, skipped


class DeepJSCC(nn.Module):
    def __init__(self, method='cnn', latent_ch=4, channel='awgn',
                 input_size=CROP_SIZE, allocation_mode=ALLOCATION_MODE):
        super().__init__()
        if allocation_mode not in ('hard', 'soft'):
            raise ValueError(f'Unknown allocation_mode: {allocation_mode}')
        assert method in ('cnn', 'semantic') + SIRA_METHODS
        self.method = method
        self.allocation_mode = allocation_mode
        self.encoder = Encoder(latent_ch)
        self.decoder = Decoder(latent_ch)
        self.channel = Channel(channel)

        # latent 空间尺寸 = input_size // 2（encoder 做一次 stride-2）
        latent_hw = (input_size // 2, input_size // 2)

        if method in SIRA_METHODS:
            self.M = SemanticPriorMapper(latent_hw=latent_hw)
            self.register_buffer(
                'vulnerability_ema',
                torch.ones(1, latent_ch, latent_hw[0], latent_hw[1]),
            )
            self.vulnerability_momentum = 0.95
            if method in SIRA_NO_R_METHODS:
                self.A = ProtectionAdapter(
                    latent_ch=latent_ch, r_dim=0,
                    allocation_mode=allocation_mode,
                )
            else:
                self.R = ChannelReliabilityMapper(embed_dim=8)
                self.A = ProtectionAdapter(
                    latent_ch=latent_ch, r_dim=8,
                    allocation_mode=allocation_mode,
                )
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

    def _snr_tensor(self, x, snr_db):
        if not torch.is_tensor(snr_db):
            snr_db = torch.full((x.shape[0],), float(snr_db), device=x.device)
        return snr_db.view(-1, 1, 1, 1).float().to(x.device)

    def _resize_vulnerability_ema(self, z):
        v = self.vulnerability_ema.to(device=z.device, dtype=z.dtype)
        if v.shape[-2:] != z.shape[-2:]:
            v = F.interpolate(v, size=z.shape[-2:], mode='area')
        return v.expand(z.shape[0], -1, -1, -1)

    def _decoder_vulnerability(self, z, x):
        if not self.training or not torch.is_grad_enabled():
            return self._resize_vulnerability_ema(z)

        with torch.enable_grad():
            z_probe = z.detach().float().requires_grad_(True)
            x_probe = x.detach().float()
            recon = self.decoder(z_probe)
            probe_loss = F.mse_loss(recon, x_probe)
            grad = torch.autograd.grad(
                probe_loss,
                z_probe,
                retain_graph=False,
                create_graph=False,
            )[0]

        v = mean_normalize(grad.detach().abs().clamp_min(1e-8), dims=(1,2,3))
        with torch.no_grad():
            v_mean = v.mean(dim=0, keepdim=True)
            if v_mean.shape[-2:] != self.vulnerability_ema.shape[-2:]:
                v_mean = F.interpolate(
                    v_mean,
                    size=self.vulnerability_ema.shape[-2:],
                    mode='area',
                )
            self.vulnerability_ema.mul_(self.vulnerability_momentum).add_(
                v_mean.to(self.vulnerability_ema.device),
                alpha=1.0 - self.vulnerability_momentum,
            )
        return v.to(dtype=z.dtype)

    def forward(self, x, snr_db):
        snr = self._snr_tensor(x, snr_db)
        z   = self.encoder(x)
        z   = power_normalize(z)

        if self.method in SIRA_METHODS:
            m, m_pix = self.M(x)
            v = self._decoder_vulnerability(z, x)
            if self.method in SIRA_NO_R_METHODS:
                z, power_symbol, risk, allocation = self.A(z, m, vulnerability=v)
            else:
                r_embed, gamma, tau = self.R(snr.view(-1))
                z, power_symbol, risk, allocation = self.A(
                    z, m, vulnerability=v, r_embed=r_embed,
                    gamma=gamma, tau=tau,
                )
            effective_power_symbol, power_alpha = effective_power_from_transmit(
                z, power_symbol,
            )
            z = power_normalize(z)
            transmit_energy = z.reshape(z.shape[0], -1).pow(2).mean(dim=1)
            power_map = power_symbol.mean(dim=1, keepdim=True)
            effective_power_map = effective_power_symbol.mean(dim=1, keepdim=True)
            self._last_sira = {
                'm': m,
                'm_pix': m_pix,
                'vulnerability': v,
                'semantic_risk': risk,
                'power_symbol': power_symbol,
                'effective_power_symbol': effective_power_symbol,
                'power_alpha': power_alpha,
                'transmit_energy': transmit_energy,
                'power_map_spatial': power_map,
                'effective_power_map_spatial': effective_power_map,
                'power_map': power_map,
                **allocation,
            }

        y = self.channel(z, snr)
        return self.decoder(y)


# ── Loss ──────────────────────────────────────────────────────

def loss_fn(model, x, x_hat, lambda_m=None):
    if model.method in SEMANTIC_LOSS_METHODS:
        w = semantic_importance(x).detach()
        if IMPORTANCE_MODE == 'dino':
            rec_loss = F.mse_loss(x_hat, x) + DINO_REC_ALPHA * (w * (x_hat - x)**2).mean()
            lambda_m = DINO_M_LAMBDA if lambda_m is None else lambda_m
        else:
            rec_loss = (w * (x_hat - x)**2).mean()
            lambda_m = 1.0 if lambda_m is None else lambda_m

        if model.method in SIRA_METHODS and hasattr(model, '_last_sira'):
            pred   = mean_normalize(model._last_sira['m_pix'], dims=(1,2,3))
            m_loss = F.l1_loss(pred, w)
            return rec_loss + lambda_m * m_loss
        return rec_loss

    return F.mse_loss(x_hat, x)
