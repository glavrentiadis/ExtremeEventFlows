from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

TensorMap = dict[str, Tensor]


def _validate_event_map(values: Mapping[str, Tensor], *, name: str) -> int:
    """Validate that every event tensor is one-dimensional and equally sized."""
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    lengths: set[int] = set()
    for key, value in values.items():
        if value.ndim != 1:
            raise ValueError(f"{name}[{key!r}] must be one-dimensional.")
        lengths.add(value.numel())

    if len(lengths) != 1:
        raise ValueError(f"All tensors in {name} must have the same length.")

    return next(iter(lengths))


@dataclass
class SourceSequence:
    """One variable-length source sequence with arbitrary event parameters."""

    values: TensorMap
    event_time: Tensor
    time_increment_name: str = "dt"

    def __post_init__(self) -> None:
        n_events = _validate_event_map(self.values, name="source values")
        if self.time_increment_name not in self.values:
            raise KeyError(
                f"Missing time-increment variable {self.time_increment_name!r}."
            )
        if self.event_time.ndim != 1 or self.event_time.numel() != n_events:
            raise ValueError("event_time must have shape [number_of_events].")

    @property
    def number_of_events(self) -> int:
        return int(self.event_time.numel())

    @property
    def dt(self) -> Tensor:
        return self.values[self.time_increment_name]

    def __getitem__(self, name: str) -> Tensor:
        return self.values[name]


@dataclass
class GroundMotionSequence:
    """Multiple ground-motion outputs and their latent residual variables."""

    values: TensorMap
    ln_values: TensorMap
    means: TensorMap
    residuals: TensorMap
    auxiliary: TensorMap

    def __post_init__(self) -> None:
        groups = {
            "ground-motion values": self.values,
            "log ground-motion values": self.ln_values,
            "ground-motion means": self.means,
            "ground-motion residuals": self.residuals,
        }
        sizes = {
            _validate_event_map(group, name=name)
            for name, group in groups.items()
            if group
        }
        if len(sizes) > 1:
            raise ValueError("All ground-motion event tensors must have equal length.")

    def __getitem__(self, name: str) -> Tensor:
        return self.values[name]


@dataclass
class SimulationSample:
    """Complete physical realization for one source sequence."""

    source: SourceSequence
    ground_motion: GroundMotionSequence


@dataclass
class TargetEvaluation:
    """Scalar target-density terms for one variable-length sequence."""

    performance: Tensor
    log_source_density: Tensor
    log_ground_motion_density: Tensor
    log_original_density: Tensor
    log_penalty: Tensor
    log_target_unnormalized: Tensor
    rare_event_indicator: Tensor


@dataclass
class FlowSequenceSample:
    """One sequence sampled from the autoregressive proposal."""

    sample: SimulationSample
    raw_events: Tensor
    base_samples: Tensor
    log_q_events: Tensor
    log_q_stop: Tensor
    log_q: Tensor

    @property
    def number_of_events(self) -> int:
        return self.sample.source.number_of_events
