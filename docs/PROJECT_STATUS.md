# Project Status and Experimental Summary

## Research Question

Can a transmitter-side power-allocation adapter improve important image
regions under a fixed transmit-power budget while reusing a pretrained
DeepJSCC encoder and decoder?

## Evidence

The [results index](../results/README.md) links to the summary and individual
runs for three experiment configurations. All use AWGN, six evaluation SNRs,
and training/evaluation seeds 42, 43, and 44. Each SIRA variant trains an
adapter against its corresponding frozen CNN backbone.

| Comparison | Operating range | DIV2K ROI | Kodak ROI | COCO object ROI |
|---|---|---:|---:|---:|
| `c=2` hard, SIRA-B2 minus B2 | −2, 0, 2 dB | +0.026 dB | −0.043 dB | — |
| `c=2` soft, SIRA-B2 minus B2 | −2, 0, 2 dB | +0.021 dB | −0.035 dB | — |
| `c=1` hard, SIRA-B2 minus CNN | All six SNRs | +0.038 dB | +0.263 dB | +0.101 dB |
| `c=1` hard, SIRA-B2 minus B2 | All six SNRs | +0.020 dB | −0.003 dB | −0.002 dB |

The entries are differences in mean ROI PSNR, averaged over the indicated SNRs.
The `c=1` DIV2K result carries the [training-split caveat](EXP3_HARD_WATERFILL_C1.md#23-datasets-and-metrics).
B2 is the semantic-weighted baseline used to initialize SIRA-B2. At `c=1`, its
Kodak and COCO ROI scores are already close to those of the full adapter.

## SNR Conditioning

The early [single-run results](../results/low_snr_awgn_c2.md) motivated a closer
look at low SNR. In the subsequent hard `c=2` experiment at −2 dB, full SIRA-B2
improved on the version without R by 0.024 dB ROI PSNR on DIV2K and fell behind
by 0.036 dB on Kodak. All three Kodak seed differences were negative. Soft
water-filling narrowed the Kodak difference to −0.023 dB at that SNR.

The [`w/o R` comparison](SIRA_NO_R_ABLATION.md) removes explicit SNR
conditioning of the allocation policy, including its SNR-dependent prior.
Both variants use the semantic risk score with decoder sensitivity. The ROI
results vary with dataset and SNR.

## Evaluation Boundaries

- The experiment compares two checkpoints of the same CNN architecture.
- The archived runs compare hard and soft priors with the learned correction
  active. A matched prior-only comparison would isolate that correction.
- The role of decoder sensitivity requires a matched comparison of semantic
  importance alone, sensitivity alone, and their product.
- The parameter count measures how much of the system is optimized. Measuring
  deployment cost requires latency, memory, and compute benchmarks.
