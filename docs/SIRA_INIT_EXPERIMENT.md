# Training DeepJSCC and SIRA-DeepJSCC

DeepJSCC trains its encoder and decoder jointly with mean squared error.
SIRA-DeepJSCC first trains an encoder and decoder with DINOv2-weighted
reconstruction, then freezes both networks and trains the transmit power
adapter from that checkpoint.

```bash
python -m scripts.train --latent_ch 1 --allocation_mode hard --importance_mode dino
```

The trainer runs the baseline, the first stage of SIRA-DeepJSCC and adapter
training in that order. The first-stage checkpoint is an input to adapter
training. The default evaluation compares the two finished models.

Existing checkpoints are reused by default. Add `--force` to retrain.
The `.pt` checkpoint records the best validation epoch, while
`_final.pt` records the last epoch. Use the same latent channel count and
allocation mode when training and evaluating.

See the [README](../README.md#training-and-evaluation) for the three-seed
evaluation command.
