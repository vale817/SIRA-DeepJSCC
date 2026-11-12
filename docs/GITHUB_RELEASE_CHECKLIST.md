# GitHub Release Checklist

Before making the repository public:

- Confirm the repository URL in `README.md`.
- Confirm the MIT license copyright holder.
- Keep datasets, DINOv2 caches, and `.pt`/`.pth` checkpoints out of Git.
- If checkpoints are shared, publish them through GitHub Releases or an external
  model host and document their hashes.
- Rerun key comparisons with the current matched-noise evaluation scripts before
  quoting final preprint numbers.
- Add the preprint URL and citation metadata when available.

Suggested repository description:

> DeepJSCC and SIRA-DeepJSCC for wireless image transmission, with semantic
> training, frozen-backbone power adaptation, and matched-seed ROI evaluation.

Suggested topics:

```text
semantic-communication deep-jscc joint-source-channel-coding pytorch
image-transmission dinov2 wireless-communication
```
