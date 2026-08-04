from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .containers import SimulationSample, TensorMap
from .ground_motion import GroundMotionModel
from .site import Site
from .source import MarkovJointSourceModel


class GroundMotionApplication(nn.Module):
    """
    Physical application composed of source, site, and ground-motion models.

    The normalizing flow is intentionally not part of this object.
    """

    def __init__(
        self,
        *,
        source_model: MarkovJointSourceModel,
        site: Site,
        ground_motion_model: GroundMotionModel,
    ) -> None:
        super().__init__()
        self.source_model = source_model
        self.site = site
        self.ground_motion_model = ground_motion_model

    def build_sample(
        self,
        *,
        source_values: Mapping[str, Tensor],
        residual_values: Mapping[str, Tensor],
    ) -> SimulationSample:
        source = self.source_model.make_sequence(source_values)
        ground_motion = self.ground_motion_model(
            source,
            self.site,
            residual_values,
        )
        return SimulationSample(source=source, ground_motion=ground_motion)

    @torch.no_grad()
    def sample_original(
        self,
        exposure_time: float,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> SimulationSample:
        source = self.source_model.sample_sequence(
            exposure_time,
            device=device,
            dtype=dtype,
        )
        residuals: TensorMap = {
            name: torch.randn(source.number_of_events, device=device, dtype=dtype)
            for name in self.ground_motion_model.residual_names
        }
        ground_motion = self.ground_motion_model(source, self.site, residuals)
        return SimulationSample(source=source, ground_motion=ground_motion)

    def log_original_density(
        self,
        sample: SimulationSample,
        exposure_time: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        log_source = self.source_model.log_prob(sample.source, exposure_time)
        log_ground_motion = self.ground_motion_model.log_prob(sample.ground_motion)
        return log_source, log_ground_motion, log_source + log_ground_motion
