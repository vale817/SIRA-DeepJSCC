# GitHub Release Checklist

Before making the repository public:

- Confirm the repository URL in `README.md`.
- Confirm the MIT license copyright holder.
- Review `Semantic Communications/SIRA_paper_outline.md`. It is an internal
  working document and is intentionally excluded from the clean release archive.
- Keep datasets, DINOv2 caches, and `.pt`/`.pth` checkpoints out of Git.
- If checkpoints are shared, publish them through GitHub Releases or an external
  model host and document their hashes.
- Rerun key comparisons with the current matched-noise evaluation scripts before
  quoting final preprint numbers.
- Add the preprint URL and citation metadata when available.

Suggested repository description:

> Parameter-efficient semantic power allocation for DeepJSCC, with low-SNR,
> object-ROI, frozen-backbone, and reliability-mapper analyses.

Suggested topics:

```text
semantic-communication deep-jscc joint-source-channel-coding pytorch
image-transmission dinov2 wireless-communication
```
