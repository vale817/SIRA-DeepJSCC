# Channel Conditioning in SIRA-DeepJSCC

The adapter uses the input SNR to compute a linear channel parameter,
an embedding and a temperature. These quantities condition its symbol-level
power allocation. Power maps can change with SNR, while the additional
reconstruction benefit from conditioning is small in the observed runs.

See the [current experiment summary](PROJECT_STATUS.md) and the
[archived diagnostic notes](https://github.com/vale817/SIRA-DeepJSCC/blob/a5cac67449d67a7a6b20b3a87a4bb412ea2b2594/docs/SIRA_NO_R_ABLATION.md).
