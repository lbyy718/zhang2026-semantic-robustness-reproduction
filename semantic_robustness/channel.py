"""Real-valued flat-fading AWGN channel from Eq. (1)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AWGNChannel(nn.Module):
    """Compute r = |h| z + w with SNR eta = |h|^2 / sigma_w^2."""

    def __init__(self, fading_gain: float = 1.0) -> None:
        super().__init__()
        if fading_gain <= 0:
            raise ValueError("fading_gain must be positive.")
        self.fading_gain = float(fading_gain)

    def noise_variance(self, snr_db: float | Tensor, reference: Tensor) -> Tensor:
        snr = torch.as_tensor(snr_db, dtype=reference.dtype, device=reference.device)
        return (self.fading_gain**2) / torch.pow(10.0, snr / 10.0)

    def forward(
        self,
        symbols: Tensor,
        snr_db: float | Tensor,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        variance = self.noise_variance(snr_db, symbols)
        while variance.ndim < symbols.ndim:
            variance = variance.unsqueeze(-1)
        if noise is None:
            noise = torch.randn(
                symbols.shape,
                dtype=symbols.dtype,
                device=symbols.device,
                generator=generator,
            ) * variance.sqrt()
        elif noise.shape != symbols.shape:
            raise ValueError("Explicit noise must have the same shape as symbols.")
        return self.fading_gain * symbols + noise


class NoiselessChannel(nn.Module):
    """Identity channel used by the R0/C0 no-noise controls."""

    def forward(
        self,
        symbols: Tensor,
        snr_db: float | Tensor | None = None,
        **_: object,
    ) -> Tensor:
        del snr_db
        return symbols
