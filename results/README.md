# Curated CIFAR-10 evidence

This directory contains small, reviewable artifacts selected from local image-transmission runs. Raw outputs, datasets, checkpoints, and console logs are excluded from Git.

- `cifar10_quick/`: clean SNR curve, PGA summary, manifests, and the generated curve from the end-to-end quick pipeline.
- `cifar10_formal/`: 1000-epoch training history and run manifest.

These artifacts support a runnable concept/trend reproduction of the image semantic-communication path. They do not establish strict numerical reproduction, certified robustness, or comparison with the traditional SSCC pipeline. See `../ASSUMPTIONS.md` for the implementation choices left open by the paper.
