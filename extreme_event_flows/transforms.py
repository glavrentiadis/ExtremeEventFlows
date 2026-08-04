from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
from torch import Tensor, nn


def inverse_softplus(y: Tensor) -> Tensor:
    """Numerically stable inverse of softplus for strictly positive y."""
    return y + torch.log(-torch.expm1(-y))


class ScalarTransform(nn.Module, ABC):
    """Monotone scalar transformation from raw to physical coordinates."""

    @abstractmethod
    def forward(self, raw: Tensor) -> Tensor:
        pass

    @abstractmethod
    def inverse(self, physical: Tensor) -> Tensor:
        pass

    @abstractmethod
    def log_abs_det_jacobian(self, raw: Tensor) -> Tensor:
        pass

    @property
    @abstractmethod
    def lower_bound(self) -> float:
        pass


class IdentityTransform(ScalarTransform):
    def forward(self, raw: Tensor) -> Tensor:
        return raw

    def inverse(self, physical: Tensor) -> Tensor:
        return physical

    def log_abs_det_jacobian(self, raw: Tensor) -> Tensor:
        return torch.zeros_like(raw)

    @property
    def lower_bound(self) -> float:
        return -math.inf


class PositiveSoftplusTransform(ScalarTransform):
    def __init__(self, minimum: float = 0.0) -> None:
        super().__init__()
        self.minimum = float(minimum)

    def forward(self, raw: Tensor) -> Tensor:
        return self.minimum + torch.nn.functional.softplus(raw)

    def inverse(self, physical: Tensor) -> Tensor:
        shifted = torch.clamp(
            physical - self.minimum,
            min=torch.finfo(physical.dtype).tiny,
        )
        return inverse_softplus(shifted)

    def log_abs_det_jacobian(self, raw: Tensor) -> Tensor:
        return torch.nn.functional.logsigmoid(raw)

    @property
    def lower_bound(self) -> float:
        return self.minimum


class BoundedSigmoidTransform(ScalarTransform):
    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        if high <= low:
            raise ValueError("high must be greater than low.")
        self.low = float(low)
        self.high = float(high)

    @property
    def width(self) -> float:
        return self.high - self.low

    def forward(self, raw: Tensor) -> Tensor:
        return self.low + self.width * torch.sigmoid(raw)

    def inverse(self, physical: Tensor) -> Tensor:
        fraction = (physical - self.low) / self.width
        eps = torch.finfo(physical.dtype).eps
        fraction = torch.clamp(fraction, min=eps, max=1.0 - eps)
        return torch.log(fraction) - torch.log1p(-fraction)

    def log_abs_det_jacobian(self, raw: Tensor) -> Tensor:
        return (
            math.log(self.width)
            + torch.nn.functional.logsigmoid(raw)
            + torch.nn.functional.logsigmoid(-raw)
        )

    @property
    def lower_bound(self) -> float:
        return self.low
