from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .containers import SimulationSample


class PerformanceFunction(nn.Module, ABC):
    """Map the complete source and multi-output ground-motion sample to a scalar."""

    @abstractmethod
    def forward(self, sample: SimulationSample) -> Tensor:
        pass


class JointGroundMotionThresholdPerformance(PerformanceFunction):
    r"""
    Joint threshold condition over multiple ground-motion outputs.

    For each event and output k, define the log margin

        r_{i,k} = log(Y_{i,k} / gamma_k).

    Event scores are:

    - ``mode='all'``: minimum margin across all outputs;
    - ``mode='any'``: maximum margin across all outputs;
    - ``mode='k_of_n'``: k-th largest margin across outputs.

    The sequence performance is the maximum event score. Consequently, the
    rare event is represented by ``performance >= 0``.
    """

    def __init__(
        self,
        thresholds: Mapping[str, float],
        *,
        mode: str = "all",
        minimum_count: int | None = None,
    ) -> None:
        super().__init__()
        if not thresholds:
            raise ValueError("thresholds cannot be empty.")
        if any(value <= 0.0 for value in thresholds.values()):
            raise ValueError("All ground-motion thresholds must be positive.")
        if mode not in {"all", "any", "k_of_n"}:
            raise ValueError("mode must be 'all', 'any', or 'k_of_n'.")

        self.names = tuple(thresholds)
        self.mode = mode
        self.minimum_count = minimum_count
        threshold_tensor = torch.tensor(
            [thresholds[name] for name in self.names],
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("thresholds", threshold_tensor)

        if mode == "k_of_n":
            if minimum_count is None:
                raise ValueError("minimum_count is required for mode='k_of_n'.")
            if not 1 <= minimum_count <= len(self.names):
                raise ValueError("minimum_count must be between 1 and n_outputs.")

    @property
    def threshold_mapping(self) -> dict[str, float]:
        """Return the current thresholds as a name-to-value dictionary."""
        values = self.thresholds.detach().cpu().tolist()
        return dict(zip(self.names, values, strict=True))

    def get_threshold(self, name: str) -> float:
        """Return the current threshold for one ground-motion output."""
        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"Unknown ground-motion output {name!r}. Available outputs: {self.names}."
            ) from exc
        return float(self.thresholds[index].detach().cpu())

    @torch.no_grad()
    def set_threshold(self, name: str, value: float) -> None:
        """Update one threshold in place without rebuilding the target object.

        Updating the registered buffer in place preserves its device and dtype.
        """
        if value <= 0.0:
            raise ValueError("A ground-motion threshold must be positive.")
        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"Unknown ground-motion output {name!r}. Available outputs: {self.names}."
            ) from exc
        self.thresholds[index].fill_(float(value))

    @torch.no_grad()
    def set_thresholds(
        self,
        thresholds: Mapping[str, float],
        *,
        require_all: bool = False,
    ) -> None:
        """Update several ground-motion thresholds in place.

        Parameters
        ----------
        thresholds
            Mapping from existing output names to new positive thresholds.
        require_all
            When ``True``, require a value for every configured output.
        """
        unknown = set(thresholds) - set(self.names)
        if unknown:
            raise KeyError(f"Unknown ground-motion outputs: {sorted(unknown)}")
        if require_all:
            missing = set(self.names) - set(thresholds)
            if missing:
                raise KeyError(f"Missing ground-motion thresholds: {sorted(missing)}")
        if any(value <= 0.0 for value in thresholds.values()):
            raise ValueError("All ground-motion thresholds must be positive.")
        for name, value in thresholds.items():
            self.set_threshold(name, value)

    def forward(self, sample: SimulationSample) -> Tensor:
        if sample.source.number_of_events == 0:
            return torch.full(
                (),
                -torch.inf,
                dtype=sample.source.event_time.dtype,
                device=sample.source.event_time.device,
            )

        missing = set(self.names) - set(sample.ground_motion.values)
        if missing:
            raise KeyError(f"Missing ground-motion outputs: {sorted(missing)}")

        value_matrix = torch.stack(
            [sample.ground_motion.values[name] for name in self.names], dim=-1
        )
        margins = torch.log(value_matrix / self.thresholds.to(value_matrix))

        if self.mode == "all":
            event_score = margins.min(dim=-1).values
        elif self.mode == "any":
            event_score = margins.max(dim=-1).values
        else:
            assert self.minimum_count is not None
            sorted_margins = torch.sort(margins, dim=-1, descending=True).values
            event_score = sorted_margins[:, self.minimum_count - 1]

        return event_score.max()


class CallablePerformanceFunction(PerformanceFunction):
    """Adapter for application-specific conditions using the full sample object."""

    def __init__(self, function) -> None:
        super().__init__()
        self.function = function

    def forward(self, sample: SimulationSample) -> Tensor:
        value = self.function(sample)
        if not torch.is_tensor(value) or value.ndim != 0:
            raise ValueError("The callable performance function must return a scalar Tensor.")
        return value
