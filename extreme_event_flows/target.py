from __future__ import annotations

import torch
from torch import Tensor, nn

from .application import GroundMotionApplication
from .containers import SimulationSample, TargetEvaluation
from .performance import PerformanceFunction


class PenaltyFunction(nn.Module):
    """Smooth rare-event penalty acting on a scalar performance value."""

    def __init__(self, *, threshold: float = 0.0, alpha: float = 35.0) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError("alpha must be positive.")
        self.threshold = float(threshold)
        self.alpha = float(alpha)

    def forward(self, performance: Tensor) -> Tensor:
        return -self.alpha * torch.relu(self.threshold - performance)


class RareEventTargetDensity(nn.Module):
    r"""Unnormalized rare-event target ``h(x) = p(x) rho(x)``."""

    def __init__(
        self,
        *,
        application: GroundMotionApplication,
        performance_function: PerformanceFunction,
        penalty_function: PenaltyFunction,
        exposure_time: float,
    ) -> None:
        super().__init__()
        if exposure_time <= 0.0:
            raise ValueError("exposure_time must be positive.")
        self.application = application
        self.performance_function = performance_function
        self.penalty_function = penalty_function
        self.exposure_time = float(exposure_time)

    def forward(self, sample: SimulationSample) -> TargetEvaluation:
        performance = self.performance_function(sample)
        log_source, log_ground_motion, log_original = (
            self.application.log_original_density(sample, self.exposure_time)
        )
        log_penalty = self.penalty_function(performance)
        return TargetEvaluation(
            performance=performance,
            log_source_density=log_source,
            log_ground_motion_density=log_ground_motion,
            log_original_density=log_original,
            log_penalty=log_penalty,
            log_target_unnormalized=log_original + log_penalty,
            rare_event_indicator=performance >= self.penalty_function.threshold,
        )
