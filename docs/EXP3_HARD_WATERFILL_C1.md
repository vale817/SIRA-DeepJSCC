# Exp3: Hard Water-Filling at `c=1`

## 1. Experiment Objective

Exp3 evaluates SIRA under the most bandwidth-constrained setting used in this
project: one latent channel (`c=1`). The experiment tests whether semantic risk
allocation remains useful when the available transmission representation is
severely compressed.

The primary comparison is between:

- a standard CNN-DeepJSCC baseline;
- a semantic-weighted DeepJSCC baseline;
- SIRA initialized from either baseline;
- full SIRA-B2 and its ablation without R.

The experiment uses hard KKT water-filling and evaluates all methods over three
random seeds.

## 2. Experimental Setup

### 2.1 Core Configuration

| Setting | Value |
|---|---|
| Experiment name | `exp3_hard_waterfill_c1` |
| Channel model | AWGN |
| Latent channels | `c=1` |
| Allocation mode | Hard KKT water-filling |
| Semantic importance | DINOv2 ViT-S/14 |
| Seeds | 42, 43, 44 |
| Training epochs | 100 per method |
| Batch size | 16 |
| Learning rate | `1e-3` |
| Training SNR range | `[-2, 15]` dB |
| Evaluation SNRs | `[-2, 0, 2, 5, 10, 15]` dB |
| Hardware | NVIDIA vGPU-32GB |

The DINOv2 repository and checkpoint were cached locally on AutoDL. Evaluation
was completed using local DINOv2 loading to avoid external GitHub dependency.

### 2.2 Compared Methods

| Identifier | Description |
|---|---|
| `cnn` | B1: CNN-DeepJSCC |
| `semantic` | B2: semantic-weighted DeepJSCC |
| `sira_b1_init` | SIRA adapter initialized from B1 |
| `sira_b2_init` | Full SIRA adapter initialized from B2 |
| `sira_b2_no_r` | SIRA-B2 without the channel reliability mapper R |

Both SIRA-B2 variants use the same form of symbol-level semantic risk:

```text
s_i = m_i * v_i
```

Here `m_i` is semantic importance and `v_i` is decoder vulnerability. Full
SIRA-B2 combines the SNR-dependent hard KKT prior with a learned correction.
The version without R retains `v_i` and uses the square-root-risk prior
without explicit SNR conditioning. Both variants normalize the scaled
symbols before transmission.

### 2.3 Datasets and Metrics

| Dataset | Evaluation role | ROI definition |
|---|---|---|
| DIV2K | In-domain reconstruction | DINOv2-derived semantic ROI |
| Kodak-24 | Out-of-domain reconstruction | DINOv2-derived semantic ROI |
| COCO val2017 | Object-focused reconstruction | Ground-truth object bounding boxes |

Metrics:

- global PSNR;
- SSIM;
- LPIPS;
- semantic ROI PSNR for DIV2K and Kodak;
- object ROI PSNR and background PSNR for COCO.

> **Important:** the current DIV2K evaluation log reports 800 images loaded
> from `DIV2K_train_HR`, despite the `DIV2K_val` JSON label. Treat this result
> as an in-domain measurement on the training split. The standard 100-image
> validation benchmark uses `DIV2K_valid_HR`.

### 2.4 Reproduction Command

```bash
python -m scripts.run_multiseed \
  --seeds 42 43 44 \
  --latent_ch 1 \
  --methods cnn semantic sira_b1_init sira_b2_init sira_b2_no_r \
  --allocation_mode hard \
  --importance_mode dino \
  --ckpt_root checkpoints/exp3_hard_waterfill_c1 \
  --result_root results/exp3_hard_waterfill_c1 \
  --eval_coco \
  --coco_root data/coco
```

## 3. Result Completeness

All three seeds completed training and evaluation on DIV2K, Kodak, and COCO.
The [archived experiment directory](../results/experiments/exp3_hard_c1/)
contains the three dataset summaries and individual runs for seeds 42, 43, and 44.

All results below use the mean over three seeds. Unless otherwise stated,
values summarized as a single number are additionally averaged over all six
evaluation SNRs.

## 4. Aggregate Results

### 4.1 DIV2K

| Method | PSNR | SSIM | LPIPS | ROI PSNR |
|---|---:|---:|---:|---:|
| CNN-DeepJSCC | 23.005 | 0.7045 | 0.4225 | 22.853 |
| Semantic-weighted | 23.023 | 0.7051 | 0.4217 | 22.871 |
| SIRA-B1-init | 23.042 | 0.7058 | 0.4214 | 22.886 |
| **SIRA-B2-init** | **23.045** | **0.7060** | **0.4210** | **22.891** |
| SIRA-B2 w/o R | 23.034 | 0.7056 | 0.4215 | 22.878 |

SIRA-B2 provides the best average result for all four reported DIV2K metrics.
Its average gains over CNN are `+0.040 dB` global PSNR and `+0.038 dB` ROI
PSNR. The full model improves ROI PSNR over the version without R by only
`+0.013 dB`.

### 4.2 Kodak-24

| Method | PSNR | SSIM | LPIPS | ROI PSNR |
|---|---:|---:|---:|---:|
| CNN-DeepJSCC | 23.061 | 0.7064 | 0.2563 | 22.615 |
| **Semantic-weighted** | **23.332** | 0.7076 | **0.2545** | **22.881** |
| SIRA-B1-init | 23.069 | 0.7066 | 0.2564 | 22.642 |
| SIRA-B2-init | 23.319 | **0.7077** | 0.2547 | 22.878 |
| SIRA-B2 w/o R | 23.317 | 0.7074 | 0.2551 | 22.873 |

SIRA-B2 improves over CNN, with average gains of `+0.258 dB`
global PSNR and `+0.263 dB` ROI PSNR. It remains approximately tied with the
semantic-weighted baseline and the version without R.

### 4.3 COCO val2017 Object ROI

| Method | PSNR | SSIM | LPIPS | Object ROI PSNR |
|---|---:|---:|---:|---:|
| CNN-DeepJSCC | 22.632 | 0.6735 | 0.2442 | 21.766 |
| **Semantic-weighted** | **22.753** | 0.6724 | 0.2431 | **21.869** |
| SIRA-B1-init | 22.624 | **0.6743** | 0.2436 | 21.771 |
| **SIRA-B2-init** | 22.744 | 0.6731 | **0.2424** | 21.867 |
| SIRA-B2 w/o R | 22.742 | 0.6728 | 0.2431 | 21.860 |

COCO provides the strongest independent semantic evaluation because its ROI
comes from ground-truth object bounding boxes rather than DINOv2. SIRA-B2
improves over CNN by `+0.112 dB` global PSNR and `+0.101 dB` object ROI PSNR on
average. It is effectively tied with the semantic-weighted baseline.

## 5. Object and Semantic ROI Results by SNR

The following tables report SIRA-B2 ROI PSNR as mean +/- standard deviation
over three seeds. Delta columns compare the mean against each reference method.

### 5.1 DIV2K Semantic ROI

| SNR | SIRA-B2 ROI PSNR | vs CNN | vs Semantic | vs w/o R |
|---:|---:|---:|---:|---:|
| -2 | 20.619 +/- 0.050 | +0.083 | +0.030 | +0.019 |
| 0 | 21.661 +/- 0.050 | +0.070 | +0.024 | +0.012 |
| 2 | 22.501 +/- 0.039 | +0.043 | +0.019 | +0.007 |
| 5 | 23.471 +/- 0.034 | +0.019 | +0.013 | +0.004 |
| 10 | 24.369 +/- 0.016 | +0.009 | +0.014 | +0.012 |
| 15 | 24.725 +/- 0.027 | +0.006 | +0.020 | +0.022 |

### 5.2 Kodak Semantic ROI

| SNR | SIRA-B2 ROI PSNR | vs CNN | vs Semantic | vs w/o R |
|---:|---:|---:|---:|---:|
| -2 | 21.113 +/- 0.151 | +0.163 | +0.001 | +0.005 |
| 0 | 22.009 +/- 0.170 | +0.193 | -0.006 | -0.000 |
| 2 | 22.691 +/- 0.194 | +0.220 | -0.009 | -0.003 |
| 5 | 23.388 +/- 0.221 | +0.277 | -0.010 | -0.001 |
| 10 | 23.937 +/- 0.261 | +0.345 | -0.003 | +0.010 |
| 15 | 24.130 +/- 0.286 | +0.380 | +0.007 | +0.021 |

### 5.3 COCO Object ROI

| SNR | SIRA-B2 Object ROI PSNR | vs CNN | vs Semantic | vs w/o R |
|---:|---:|---:|---:|---:|
| -2 | 20.058 +/- 0.075 | +0.058 | +0.007 | +0.009 |
| 0 | 20.949 +/- 0.088 | +0.062 | -0.003 | +0.001 |
| 2 | 21.643 +/- 0.106 | +0.075 | -0.010 | -0.002 |
| 5 | 22.371 +/- 0.139 | +0.104 | -0.011 | +0.000 |
| 10 | 22.981 +/- 0.180 | +0.144 | -0.003 | +0.012 |
| 15 | 23.198 +/- 0.200 | +0.165 | +0.006 | +0.022 |

## 6. Main Findings

### 6.1 SIRA-B2 consistently improves over CNN under `c=1`

Across the six-SNR averages, SIRA-B2 improves ROI PSNR over CNN by:

| Dataset | ROI PSNR gain |
|---|---:|
| DIV2K | +0.038 dB |
| Kodak-24 | +0.263 dB |
| COCO object ROI | +0.101 dB |

The improvement is most substantial on Kodak and remains visible on COCO
ground-truth object regions.

### 6.2 Semantic initialization is important

SIRA-B2 is consistently stronger than SIRA-B1 on Kodak and COCO. This indicates
that the quality of the frozen backbone remains a major source of performance.
The adapter does not compensate for a substantially weaker backbone by itself.

### 6.3 Full SIRA is competitive with, but does not consistently beat, the
semantic baseline

SIRA-B2 slightly exceeds the semantic baseline on DIV2K, but is approximately
tied or slightly below it on Kodak and COCO. These results support a
"competitive with semantic weighting" claim, not a universal improvement claim.

### 6.4 The average differences from the version without R are small

Average ROI PSNR gains of full SIRA-B2 over `SIRA-B2 w/o R` are:

| Dataset | Full minus w/o R |
|---|---:|
| DIV2K | +0.013 dB |
| Kodak-24 | +0.005 dB |
| COCO object ROI | +0.007 dB |

These differences vary across SNRs and seeds. The ablation measures the
combined effect of explicit SNR conditioning and the SNR-dependent prior.
Decoder vulnerability remains active in both conditions, so its contribution
requires a separate matched ablation.

## 7. Recommended Paper Usage

### Main Paper

Use:

1. a compact table comparing CNN, Semantic, SIRA-B2, and SIRA-B2 w/o R;
2. a COCO object ROI PSNR-versus-SNR figure using three-seed means and error
   bands;
3. `c=1` as an extreme-bandwidth stress test alongside the main `c=2` results.

The strongest supported statement is:

> Under the severe bandwidth constraint of `c=1`, SIRA-B2 consistently
> improves over CNN-DeepJSCC and remains competitive with a
> semantic-weighted backbone, including on independently annotated COCO object
> regions.

### Ablation or Appendix

Place the following in the ablation section or appendix:

- full SIRA-B2 versus SIRA-B2 w/o R;
- SIRA-B1 versus SIRA-B2 initialization;
- full DIV2K and Kodak PSNR/SSIM/LPIPS curves;
- individual-seed results.

### Claims Not Supported by Exp3

Exp3 does not support the following claims:

- SIRA universally outperforms the semantic-weighted baseline;
- the decoder vulnerability term produces a large or statistically significant
  reconstruction gain;
- the current DIV2K numbers are standard DIV2K validation-set results;
- SIRA is state of the art.

## 8. Suggested Paper Wording

> Under the severe bandwidth constraint of `c=1`, SIRA-B2 improves average ROI
> PSNR over CNN-DeepJSCC by 0.263 dB on Kodak and 0.101 dB on COCO object
> regions, while remaining competitive with the semantic-weighted baseline.
> The comparison with the version without R yields small,
> condition-dependent differences in ROI PSNR.

