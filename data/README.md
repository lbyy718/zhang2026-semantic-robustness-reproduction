# Data preparation

Datasets are not committed to this repository. Keep the following local layout:

```text
data/
├── cifar10/
└── cost2100/
    └── original/
        ├── DATA_Htrainout.mat
        ├── DATA_Hvalout.mat
        └── DATA_Htestout.mat
```

## CIFAR-10

The image configurations use `torchvision.datasets.CIFAR10` with downloading
enabled, so the first training or evaluation run prepares the dataset under
`data/cifar10/` automatically.

Expected original archive checksum:

```text
MD5 c58f30108f718f92721af3b95e74349a
```

## COST2100

The CSI experiments use the public outdoor COST2100 files distributed with the
CsiNet reproduction data:

```text
DATA_Htrainout.mat  HT[100000, 2048]
DATA_Hvalout.mat    HT[30000, 2048]
DATA_Htestout.mat   HT[20000, 2048]
```

Place the three files under `data/cost2100/original/`. The loader reconstructs
the stored real/imaginary planes, applies the representation selected by the
JSON configuration, and reuses the training-set normalization bounds for
validation and test data.

The original data licenses and redistribution terms apply. Do not commit the
multi-gigabyte archives or `.mat` files to Git.
