# Configuration guide

All JSON paths are resolved relative to the `configs/` directory.

| Configuration | Purpose |
| --- | --- |
| `image_cifar10.json` | Formal 1000-epoch CIFAR-10 DeepJSCC training, clean evaluation, and attacks |
| `image_cifar10_quick.json` | Small end-to-end pipeline check |

The quick configuration verifies that the complete pipeline runs. Its metrics must not be presented as formal paper-reproduction numbers.

To create an ablation, copy one of these files and change only the factor under test. Keep a separate output directory so checkpoints and CSV files are never overwritten.
