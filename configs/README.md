# Configuration guide

All paths in a JSON file are resolved relative to the `configs/` directory.

## Main experiments

| Configuration | Purpose |
| --- | --- |
| `image_cifar10.json` | Formal 1000-epoch CIFAR-10 DeepJSCC training and evaluation |
| `csi_cost2100.json` | Paper-shaped 1000-epoch single-channel COST2100 baseline |

## Quick checks

| Configuration | Purpose |
| --- | --- |
| `image_cifar10_quick.json` | Small CIFAR-10 end-to-end train/clean/PGA check |
| `csi_cost2100_quick.json` | Small COST2100 end-to-end train/clean/PGA check |

Quick configurations prove that the complete pipeline runs, but their metrics
must not be presented as formal paper reproduction numbers.

## COST2100 ablations

Files beginning with `csi_cost2100_ablation_` record the staged investigation
of CSI representation, loss, decoder nonlinearities, convolution kernels,
complex channel-use accounting, global bottleneck mixing, and weight decay.

The strongest retained configuration is:

```text
csi_cost2100_ablation_single_channel_mse_complex256_kernel5_global_mixing_200.json
```

It is kept together with the unsuccessful intermediate configurations so that
the reported ablation path remains auditable rather than showing only the best
run.
