# Experiment Results

## Three-seed experiments

The archived runs use seeds 42, 43, and 44 and evaluate AWGN at −2, 0, 2, 5,
10, and 15 dB. Each experiment directory contains `summary_*.json` files
with means and standard deviations across seeds and `seed_42/`, `seed_43/`,
and `seed_44/` with the individual metric arrays. The JSON files retain their
original dataset and method labels.

| Experiment | Latent channels | Allocation | Summary files | Individual runs |
|---|---:|---|---|---|
| Exp1 | 2 | Hard water-filling | [DIV2K](experiments/exp1_hard_c2/summary_div2k_awgn_c2.json), [Kodak-24](experiments/exp1_hard_c2/summary_kodak_awgn_c2.json) | [42](experiments/exp1_hard_c2/seed_42/), [43](experiments/exp1_hard_c2/seed_43/), [44](experiments/exp1_hard_c2/seed_44/) |
| Exp2 | 2 | Soft water-filling | [DIV2K](experiments/exp2_soft_c2/summary_div2k_awgn_c2.json), [Kodak-24](experiments/exp2_soft_c2/summary_kodak_awgn_c2.json) | [42](experiments/exp2_soft_c2/seed_42/), [43](experiments/exp2_soft_c2/seed_43/), [44](experiments/exp2_soft_c2/seed_44/) |
| Exp3 | 1 | Hard water-filling | [DIV2K](experiments/exp3_hard_c1/summary_div2k_awgn_c1.json), [Kodak-24](experiments/exp3_hard_c1/summary_kodak_awgn_c1.json), [COCO val2017](experiments/exp3_hard_c1/summary_coco_awgn_c1.json) | [42](experiments/exp3_hard_c1/seed_42/), [43](experiments/exp3_hard_c1/seed_43/), [44](experiments/exp3_hard_c1/seed_44/) |

DIV2K and Kodak use DINOv2-derived semantic regions. COCO uses annotated
object boxes. For Exp3, the archived run notes report 800 images from
`DIV2K_train_HR` despite the `DIV2K_val` JSON label. Treat that result as
an in-domain measurement on the training split. The [Exp3 report](../docs/EXP3_HARD_WATERFILL_C1.md)
contains the reproduction command and the split caveat.

The `sira_b2_no_r` condition retains decoder sensitivity and removes explicit
SNR conditioning of the allocation policy. See the [ablation notes](../docs/SIRA_NO_R_ABLATION.md)
for the role of the SNR-dependent prior in that comparison.

## Historical single runs

The top-level `*_awgn_c2.*`, `*_awgn_c4.*`, and `low_snr_*` files record
exploratory runs. Some predate matched-noise evaluation. The legacy `sira`
label denotes a B1-initialized adapter. Treat these measurements separately
from the three-seed comparisons above.

The [supplemental Kodak figure](qualitative/README.md) comes from a separate
B1 checkpoint and illustrates local detail loss in baseline reconstruction.

For new runs, use a separate output directory:

```bash
python -m scripts.eval --result_dir results/run_example ...
```
