# Evaluation Results

The main result compares **DeepJSCC** and **SIRA-DeepJSCC** under AWGN
with one latent channel and hard water-filling.

- [Main ROI-PSNR figure](sira_main_roi_c1.png)
- [Paired measurements by dataset, SNR and seed](sira_main_roi_c1.csv)
- [Plotting script](../scripts/plot_main_roi.py)

In the CSV, `cnn_deepjscc_roi_psnr_db` is the DeepJSCC baseline and
`sira_roi_psnr_db` is the complete SIRA-DeepJSCC model. Kodak-24 uses a
semantic ROI, while COCO val2017 uses annotated object regions.

The other JSON, CSV, PNG and PDF files are historical exploratory runs at
`c=2` and `c=4`. They retain their original method identifiers for
reproducibility.
