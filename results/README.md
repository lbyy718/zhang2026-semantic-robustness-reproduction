# Curated reproduction evidence

This directory contains small, reviewable artifacts selected from local runs.
Raw outputs, datasets, intermediate checkpoints, and console logs are excluded
from Git.

## Included evidence

- `cifar10_quick/`: clean SNR curve and PGA attack summary from the end-to-end
  quick pipeline.
- `cifar10_formal/`: 1000-epoch training history and run manifest.
- `cost2100_formal/`: clean curve and manifest for the formal baseline.
- `cost2100_ablation/`: five-repeat baseline/best-model comparison, plot,
  manifests, and the detailed interpretation of the roughly 3 dB improvement.

## Interpretation boundary

These artifacts support a runnable concept/trend reproduction of the semantic
communication side. They do not establish strict numerical reproduction of all
paper curves, and they do not include the traditional SSCC pipeline. See
`../ASSUMPTIONS.md` for the implementation choices that the paper leaves open.
