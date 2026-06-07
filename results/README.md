# Included Results

This directory contains representative exploratory results used during SIRA
development.

- `*_awgn_c2.*`: constrained-bandwidth experiments with latent width `c=2`.
- `*_awgn_c4.*`: wider latent experiments with latent width `c=4`.
- `low_snr_*`: focused summaries for `-2`, `0`, and `2` dB.

The JSON files currently contain the historical `cnn`, `semantic`, and legacy
`sira` runs. In the code, legacy `sira` is equivalent in configuration to
`sira_b1_init`, but independently trained checkpoints can produce different
numbers.

Evaluation samples random channel noise. The included measurements are useful
for project inspection and figure reproduction, but publication-quality
reporting should repeat evaluation with multiple seeds and confidence intervals.
Some tracked historical results predate the matched-noise evaluation seed added
to the current evaluation scripts.

New experiment outputs should be written to a named subdirectory:

```bash
python eval.py --result_dir results/my_run ...
```
