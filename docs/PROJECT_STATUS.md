# DeepJSCC and SIRA-DeepJSCC Experiments

The project evaluates semantic-aware image transmission under fixed transmit
power. **DeepJSCC** is the CNN baseline, trained end to end with mean squared
error. **SIRA-DeepJSCC** first trains a backbone with DINOv2-weighted
reconstruction, then freezes the encoder and decoder and trains a transmit
power adapter.

## Main comparison

The main experiment uses AWGN, one latent channel and hard water-filling.
Each model is evaluated with seeds 42–44 at SNRs of −2, 0, 2, 5, 10 and 15 dB.
Kodak-24 uses DINOv2-derived semantic regions. COCO val2017 uses object regions
defined by annotated bounding boxes. The archived DIV2K run at this bandwidth
evaluated the training split.

SIRA-DeepJSCC has higher mean ROI PSNR than DeepJSCC at every evaluated SNR on
the two test datasets shown. The [main figure](../README.md#main-result)
reports the mean and standard deviation across three seeds alongside paired
differences. The [per-seed measurements](../results/sira_main_roi_c1.csv)
provide the underlying data.

## Interpretation

The comparison reflects the combined effects of semantic training and power
adaptation. Semantic training appears to account for much of the observed
improvement. Reusing DINOv2-derived importance in both stages may limit the
additional information available to the adapter. The evidence is specific to
the tested CNN architecture, bandwidth, channel and datasets.
