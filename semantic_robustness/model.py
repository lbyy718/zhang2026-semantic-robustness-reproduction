"""DeepJSCC image encoder/decoder reconstructed from Table I of the paper.

The paper specifies tensor sizes but omits kernel sizes and the residual-block
internals. This implementation uses configurable odd kernels and a two-layer
residual block; the exact choices are recorded in the JSON configuration.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class ResidualBlock(nn.Module):
    """A shape-preserving two-convolution residual block."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Residual-block kernel_size must be odd.")
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.body(inputs))


class PowerNormalizer(nn.Module):
    """Normalize every codeword to the paper constraint ||z||_2^2 = N."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, symbols: Tensor) -> Tensor:
        flat = symbols.flatten(start_dim=1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(self.eps)
        return flat * (math.sqrt(flat.shape[1]) / norm)


class DeepJSCCEncoder(nn.Module):
    """Table-I image encoder with two stride-2 convolutions."""

    def __init__(
        self,
        in_channels: int,
        channel_multiplier: int,
        spatial_size: tuple[int, int] = (32, 32),
        kernel_size: int = 3,
        residual_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Encoder kernel_size must be odd.")
        if spatial_size[0] % 4 or spatial_size[1] % 4:
            raise ValueError("Encoder spatial_size must be divisible by four.")

        latent_channels = 2 * channel_multiplier
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size, stride=2, padding=padding),
            nn.ReLU(inplace=True),
            ResidualBlock(16, residual_kernel_size),
            nn.Conv2d(16, latent_channels, kernel_size, stride=2, padding=padding),
            nn.ReLU(inplace=True),
            ResidualBlock(latent_channels, residual_kernel_size),
        )
        self.normalizer = PowerNormalizer()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.normalizer(self.network(inputs))


class DeepJSCCDecoder(nn.Module):
    """Table-I image decoder accepting flattened real channel symbols."""

    def __init__(
        self,
        out_channels: int,
        channel_multiplier: int,
        spatial_size: tuple[int, int] = (32, 32),
        kernel_size: int = 3,
        residual_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        height, width = spatial_size
        if height % 4 or width % 4:
            raise ValueError("spatial_size must be divisible by four.")
        if kernel_size % 2 == 0:
            raise ValueError("Decoder kernel_size must be odd.")

        self.latent_shape = (2 * channel_multiplier, height // 4, width // 4)
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.ConvTranspose2d(self.latent_shape[0], 32, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size, padding=padding),
            nn.Sigmoid(),
            ResidualBlock(32, residual_kernel_size),
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            ResidualBlock(16, residual_kernel_size),
            nn.ConvTranspose2d(
                16,
                out_channels,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    @property
    def channel_uses(self) -> int:
        return math.prod(self.latent_shape)

    def forward(self, received: Tensor) -> Tensor:
        if received.ndim == 2:
            if received.shape[1] != self.channel_uses:
                raise ValueError(
                    f"Expected {self.channel_uses} flattened channel symbols, "
                    f"received {received.shape[1]}."
                )
            received = received.reshape(received.shape[0], *self.latent_shape)
        elif tuple(received.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Expected latent shape {self.latent_shape}, "
                f"got {tuple(received.shape[1:])}."
            )
        return self.network(received)


class DeepJSCC(nn.Module):
    """End-to-end DeepJSCC model for CIFAR-10 image transmission."""

    def __init__(
        self,
        in_channels: int = 3,
        channel_multiplier: int = 6,
        spatial_size: tuple[int, int] = (32, 32),
        kernel_size: int = 3,
        residual_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.encoder = DeepJSCCEncoder(
            in_channels,
            channel_multiplier,
            spatial_size=spatial_size,
            kernel_size=kernel_size,
            residual_kernel_size=residual_kernel_size,
        )
        self.decoder = DeepJSCCDecoder(
            in_channels,
            channel_multiplier,
            spatial_size=spatial_size,
            kernel_size=kernel_size,
            residual_kernel_size=residual_kernel_size,
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def channel_uses(self) -> int:
        return self.decoder.channel_uses

    @property
    def source_dimension(self) -> int:
        return self.in_channels * math.prod(self.spatial_size)

    @property
    def bandwidth_ratio(self) -> float:
        return self.channel_uses / self.source_dimension

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, received: Tensor) -> Tensor:
        return self.decoder(received)

    def forward(self, inputs: Tensor, channel: nn.Module, snr_db: float | Tensor) -> Tensor:
        return self.decode(channel(self.encode(inputs), snr_db))


class ClassificationHead(nn.Module):
    """Parameter-matched classifier operating on the same 768-symbol codeword."""

    def __init__(self, channel_uses: int, hidden_features: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(channel_uses, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, received: Tensor) -> Tensor:
        return self.network(received.flatten(start_dim=1))


class DeepJSCCClassifier(nn.Module):
    """Classification control sharing the exact DeepJSCC encoder and channel code."""

    def __init__(
        self,
        in_channels: int = 3,
        channel_multiplier: int = 6,
        spatial_size: tuple[int, int] = (32, 32),
        kernel_size: int = 3,
        residual_kernel_size: int = 3,
        classifier_hidden: int = 53,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.encoder = DeepJSCCEncoder(
            in_channels,
            channel_multiplier,
            spatial_size=spatial_size,
            kernel_size=kernel_size,
            residual_kernel_size=residual_kernel_size,
        )
        channel_uses = 2 * channel_multiplier * (spatial_size[0] // 4) * (
            spatial_size[1] // 4
        )
        self._channel_uses = channel_uses
        # Keep the decoder attribute so latent-space attacks use the same interface.
        self.decoder = ClassificationHead(
            channel_uses, classifier_hidden, num_classes
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def channel_uses(self) -> int:
        return self._channel_uses

    @property
    def source_dimension(self) -> int:
        return self.in_channels * math.prod(self.spatial_size)

    @property
    def bandwidth_ratio(self) -> float:
        return self.channel_uses / self.source_dimension

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, received: Tensor) -> Tensor:
        return self.decoder(received)

    def forward(self, inputs: Tensor, channel: nn.Module, snr_db: float | Tensor) -> Tensor:
        return self.decode(channel(self.encode(inputs), snr_db))


class CIFAR10InputNormalizer(nn.Module):
    """Differentiable CIFAR-10 normalization for models receiving [0, 1] images."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(CIFAR10_MEAN).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CIFAR10_STD).reshape(1, 3, 1, 1)
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "CIFAR10InputNormalizer expects images shaped [batch, 3, H, W]."
            )
        return (images - self.mean) / self.std


class ResNet18BottleneckEncoder(nn.Module):
    """CIFAR ResNet-18 encoder followed by a unit-power 768-symbol bottleneck."""

    def __init__(self, latent_dim: int = 768, feature_dim: int = 512) -> None:
        super().__init__()
        if latent_dim <= 0 or feature_dim != 512:
            raise ValueError("ResNet-18 requires latent_dim > 0 and feature_dim=512.")
        try:
            from torchvision.models import resnet18
        except ImportError as exc:  # pragma: no cover - explicit dependency error
            raise ImportError("resnet18_bottleneck requires torchvision.") from exc

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.normalizer = CIFAR10InputNormalizer()
        self.backbone = backbone
        self.projection = nn.Linear(feature_dim, latent_dim)
        self.power_normalizer = PowerNormalizer()
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity="linear")
        nn.init.zeros_(self.projection.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.backbone(self.normalizer(inputs))
        return self.power_normalizer(self.projection(features))


class DeepJSCCResNetClassifier(nn.Module):
    """Strong classification control with an explicit semantic channel bottleneck.

    The model is trained from scratch.  It is deliberately not the frozen
    evaluator used by the reconstruction endpoint: only the CIFAR-style
    ResNet-18 architecture is shared.
    """

    architecture = "resnet18_bottleneck"

    def __init__(
        self,
        in_channels: int = 3,
        spatial_size: tuple[int, int] = (32, 32),
        latent_dim: int = 768,
        feature_dim: int = 512,
        classifier_hidden: int = 512,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if in_channels != 3 or tuple(spatial_size) != (32, 32):
            raise ValueError(
                "resnet18_bottleneck currently supports 3x32x32 CIFAR images only."
            )
        if classifier_hidden <= 0 or num_classes < 2:
            raise ValueError("classifier_hidden must be positive and num_classes >= 2.")
        self.in_channels = in_channels
        self.spatial_size = tuple(spatial_size)
        self._channel_uses = int(latent_dim)
        self.encoder = ResNet18BottleneckEncoder(
            latent_dim=self._channel_uses, feature_dim=feature_dim
        )
        self.decoder = ClassificationHead(
            self._channel_uses, int(classifier_hidden), int(num_classes)
        )
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def channel_uses(self) -> int:
        return self._channel_uses

    @property
    def source_dimension(self) -> int:
        return self.in_channels * math.prod(self.spatial_size)

    @property
    def bandwidth_ratio(self) -> float:
        return self.channel_uses / self.source_dimension

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, received: Tensor) -> Tensor:
        return self.decoder(received)

    def forward(self, inputs: Tensor, channel: nn.Module, snr_db: float | Tensor) -> Tensor:
        return self.decode(channel(self.encode(inputs), snr_db))
