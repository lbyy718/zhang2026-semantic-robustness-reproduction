# Configuration guide

## C-strong three-arm controls

`configs/cstrong/` contains the locked strong-classifier controls:

- `cs0_resnet18_noiseless_seed2026.json`: no training AWGN and no Jacobian penalty;
- `cs1_resnet18_awgn10_seed2026.json`: 10 dB latent AWGN training;
- `csj_resnet18_jacobian_seed2026.json`: noiseless training with the latent failure-score Jacobian penalty.

Use `scripts/run_cstrong_pilot.py`; do not edit one arm independently unless the same non-treatment change is applied to all three.

All JSON paths are resolved relative to the `configs/` directory.

| Configuration | Purpose |
| --- | --- |
| `image_cifar10.json` | Formal 1000-epoch CIFAR-10 DeepJSCC training, clean evaluation, and attacks |
| `image_cifar10_quick.json` | Small end-to-end pipeline check |
| `image_cifar10_seed2027.json` | Second legacy R1 run; only the training seed changes from 2026 to 2027 |
| `factorial/r0_reconstruction_noiseless_seed2026.json` | R0: reconstruction, truly noiseless training |
| `factorial/r1_reconstruction_awgn10_seed2026.json` | R1: reconstruction, 10 dB AWGN training |
| `factorial/c0_classification_noiseless_seed2026.json` | C0: classification, truly noiseless training |
| `factorial/c1_classification_awgn10_seed2026.json` | C1: classification, 10 dB AWGN training |

The quick configuration verifies that the complete pipeline runs. Its metrics must not be presented as formal paper-reproduction numbers.

To create an ablation, copy one of these files and change only the factor under test. Keep a separate output directory so checkpoints and CSV files are never overwritten.

The four factorial configurations are directly runnable. `training.channel_noise=false`
selects an exact identity channel during training; a very large finite SNR is not
used as a silent substitute. All four models are still evaluated through the same
AWGN channel over the configured SNR grid.

The legacy `image_cifar10*.json` runs use the CIFAR-10 test split during checkpoint
selection so that seed 2027 is directly comparable with the completed seed-2026
baseline. The confirmatory factorial configurations instead use one fixed 45k/5k
train/validation split and reserve the test split for final evaluation.

For exact switching commands and interpretation, see
`../docs/06_factorial_config_guide.md`. For the complete follow-up order, see
`../docs/04_formal_training_evaluation.md` and `../docs/05_factorial_experiment_plan.md`.
The resumable 12-run launcher and result layout are documented in
`../docs/07_factorial_batch_run.md`.
