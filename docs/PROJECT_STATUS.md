# Project Status and Experimental Summary

## Research Question

Can a lightweight, frozen-backbone adapter improve semantically important image
regions under low-bandwidth and noisy-channel conditions?

## Completed Experiments

1. Trained CNN-DeepJSCC (`cnn`) and semantic-weighted DeepJSCC (`semantic`) at
   latent channel widths `c=2` and `c=4`.
2. Trained SIRA from a CNN backbone (`sira_b1_init`) and a semantic backbone
   (`sira_b2_init`) while freezing encoder and decoder parameters.
3. Evaluated DIV2K, Kodak-24, and COCO val2017 object ROIs across six AWGN SNRs.
4. Compared global PSNR, SSIM, LPIPS, semantic ROI PSNR, and object-ROI PSNR.
5. Produced low-SNR summaries and power-map visualizations.
6. Ablated the reliability mapper R using `sira_b2_no_r`.
7. Measured whether R actually changes the power map across SNR conditions.

## Current Evidence

### Bandwidth matters

At `c=2`, semantic methods show clearer differences, especially at low SNR.
At `c=4`, method differences are generally small. Semantic protection therefore
appears most relevant when transmission capacity is constrained.

### Frozen-backbone quality matters

`sira_b1_init` and `sira_b2_init` use the same adapter structure but different
frozen checkpoints. Their behavior differs, showing that the adapter cannot be
understood independently of the representation learned by the backbone.

This supports the claim that SIRA is compatible with multiple pretrained
DeepJSCC checkpoints. It does not establish architecture-agnostic behavior.

### Semantic gains can saturate

The semantic backbone and M both use DINOv2-derived information.
`sira_b2_init` only slightly changes the performance of the semantic backbone,
which is consistent with diminishing marginal returns once the backbone already
captures the same semantic prior.

This is an interpretation supported by the current evidence, not a fully
isolated causal conclusion.

### R changes behavior but adds little measured utility

The `sira_b2_no_r` ablation performs nearly identically to Full SIRA-B2.
Sensitivity analysis shows that Full SIRA-B2 changes its spatial power pattern
with SNR, so R is not simply ignored. However:

- the relative power-map changes remain small;
- allocation is close to uniform;
- concentration does not consistently increase at low SNR;
- the behavior does not translate into stable reconstruction gains.

The current measurable benefit is therefore primarily associated with semantic
spatial allocation through M+A, while explicit reliability conditioning remains
an open design problem.

Representative single-run Full-minus-w/o-R differences:

| Dataset | Metric | Low-SNR mean (`-2, 0, 2` dB) | All-SNR mean |
|---|---:|---:|---:|
| DIV2K | PSNR | -0.005 dB | -0.005 dB |
| DIV2K | ROI PSNR | +0.009 dB | -0.001 dB |
| Kodak-24 | PSNR | -0.025 dB | -0.001 dB |
| Kodak-24 | ROI PSNR | -0.031 dB | +0.004 dB |

These differences are too small and inconsistent to support a stable R-module
gain claim.

## Claims Appropriate for a Preprint or Portfolio

- A parameter-efficient semantic power-allocation adapter was implemented and
  evaluated across in-domain, out-of-domain, and object-ROI settings.
- Semantic-aware protection is most useful under constrained bandwidth and low
  SNR in this experimental setup.
- Adapter behavior depends on frozen-backbone semantic quality.
- Explicit SNR conditioning does not automatically produce a useful
  low-SNR-focused allocation policy.

## Claims Not Supported Yet

- Universal improvement over DeepJSCC baselines.
- Architecture-agnostic adaptation.
- State-of-the-art semantic communication performance.
- A causal proof that duplicated DINOv2 supervision alone causes semantic
  saturation.

## Recommended Next Steps

For a portfolio/preprint release, prioritize documentation, repeatability, and
clear limitations over additional experiments. If research resumes, the most
useful additions would be repeated evaluations with confidence intervals and an
objective that directly rewards low-SNR ROI preservation.
