from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor, nn

from .containers import GroundMotionSequence, SourceSequence, TensorMap
from .site import Site


class GroundMotionFeatureBuilder(nn.Module, ABC):
    """Build arbitrary event-by-feature inputs for a ground-motion model."""

    @property
    @abstractmethod
    def number_of_features(self) -> int:
        pass

    @abstractmethod
    def forward(self, source: SourceSequence, site: Site) -> Tensor:
        """Return a tensor with shape [number_of_events, number_of_features]."""


class PointSourceFeatureBuilder(GroundMotionFeatureBuilder):
    """
    Example feature builder using magnitude, 3-D source-to-site distance,
    Vs30, and optional additional source/site scalar fields.
    """

    def __init__(
        self,
        *,
        magnitude_name: str = "magnitude",
        source_location_names: tuple[str, str, str] = ("x", "y", "depth"),
        site_location_names: tuple[str, str, str] = ("x", "y", "depth"),
        vs30_name: str = "vs30",
        additional_source_fields: Sequence[str] = (),
        additional_site_fields: Sequence[str] = (),
        distance_offset: float = 5.0,
        reference_vs30: float = 500.0,
    ) -> None:
        super().__init__()
        if distance_offset <= 0.0 or reference_vs30 <= 0.0:
            raise ValueError("distance_offset and reference_vs30 must be positive.")
        self.magnitude_name = magnitude_name
        self.source_location_names = source_location_names
        self.site_location_names = site_location_names
        self.vs30_name = vs30_name
        self.additional_source_fields = tuple(additional_source_fields)
        self.additional_site_fields = tuple(additional_site_fields)
        self.distance_offset = float(distance_offset)
        self.reference_vs30 = float(reference_vs30)

    @property
    def number_of_features(self) -> int:
        # bias, magnitude, log distance, log Vs30, plus optional fields
        return 4 + len(self.additional_source_fields) + len(self.additional_site_fields)

    def forward(self, source: SourceSequence, site: Site) -> Tensor:
        n_events = source.number_of_events
        if n_events == 0:
            return torch.empty(
                (0, self.number_of_features),
                dtype=source.event_time.dtype,
                device=source.event_time.device,
            )

        source_location = torch.stack(
            [source[name] for name in self.source_location_names], dim=-1
        )
        site_location = torch.stack(
            [site.field(name) for name in self.site_location_names]
        ).to(source_location)
        distance = torch.linalg.vector_norm(source_location - site_location, dim=-1)
        vs30 = site.field(self.vs30_name).to(source.event_time)

        features = [
            torch.ones_like(source.event_time),
            source[self.magnitude_name],
            torch.log(distance + self.distance_offset),
            torch.log(vs30 / self.reference_vs30).expand(n_events),
        ]
        features.extend(source[name] for name in self.additional_source_fields)
        features.extend(
            site.field(name).to(source.event_time).expand(n_events)
            for name in self.additional_site_fields
        )
        return torch.stack(features, dim=-1)


class GroundMotionModel(nn.Module, ABC):
    """Ground-motion model with arbitrary source inputs and multiple outputs."""

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]:
        pass

    @property
    @abstractmethod
    def residual_names(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def forward(
        self,
        source: SourceSequence,
        site: Site,
        residuals: Mapping[str, Tensor],
    ) -> GroundMotionSequence:
        pass

    @abstractmethod
    def log_prob(self, ground_motion: GroundMotionSequence) -> Tensor:
        pass


class MultiOutputGaussianGroundMotionModel(GroundMotionModel):
    r"""
    Configurable multi-output lognormal ground-motion model.

    For feature vector ``f_i`` and output ``k``,

        mu_i = B f_i,
        ln Y_i = mu_i + L epsilon_i,
        epsilon_i ~ N(0, I).

    ``L`` permits correlated ground-motion outputs within each event. The
    number of source inputs is controlled entirely by the feature builder, and
    the number of ground-motion outputs is controlled by ``output_names``.
    """

    def __init__(
        self,
        *,
        feature_builder: GroundMotionFeatureBuilder,
        output_names: Sequence[str],
        coefficients: Tensor | Sequence[Sequence[float]],
        residual_cholesky: Tensor | Sequence[Sequence[float]],
        residual_prefix: str = "epsilon_",
    ) -> None:
        super().__init__()
        self.feature_builder = feature_builder
        self._output_names = tuple(output_names)
        if not self._output_names:
            raise ValueError("output_names cannot be empty.")
        if len(set(self._output_names)) != len(self._output_names):
            raise ValueError("output_names must be unique.")
        self._residual_names = tuple(
            f"{residual_prefix}{name}" for name in self._output_names
        )

        coefficient_tensor = torch.as_tensor(
            coefficients, dtype=torch.get_default_dtype()
        )
        cholesky_tensor = torch.as_tensor(
            residual_cholesky, dtype=torch.get_default_dtype()
        )
        n_outputs = len(self._output_names)
        if coefficient_tensor.shape != (
            n_outputs,
            feature_builder.number_of_features,
        ):
            raise ValueError(
                "coefficients must have shape "
                f"[{n_outputs}, {feature_builder.number_of_features}]."
            )
        if cholesky_tensor.shape != (n_outputs, n_outputs):
            raise ValueError(
                f"residual_cholesky must have shape [{n_outputs}, {n_outputs}]."
            )
        if not torch.allclose(cholesky_tensor, torch.tril(cholesky_tensor)):
            raise ValueError("residual_cholesky must be lower triangular.")
        if torch.any(torch.diagonal(cholesky_tensor) <= 0.0):
            raise ValueError("residual_cholesky must have positive diagonal.")

        self.register_buffer("coefficients", coefficient_tensor)
        self.register_buffer("residual_cholesky", cholesky_tensor)

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def residual_names(self) -> tuple[str, ...]:
        return self._residual_names

    def forward(
        self,
        source: SourceSequence,
        site: Site,
        residuals: Mapping[str, Tensor],
    ) -> GroundMotionSequence:
        missing = set(self.residual_names) - set(residuals)
        if missing:
            raise KeyError(f"Missing ground-motion residuals: {sorted(missing)}")

        features = self.feature_builder(source, site)
        n_events = source.number_of_events
        epsilon = torch.stack([residuals[name] for name in self.residual_names], dim=-1)
        if epsilon.shape != (n_events, len(self.output_names)):
            raise ValueError("Residual tensors have inconsistent event dimensions.")

        mean_matrix = features @ self.coefficients.T
        ln_value_matrix = mean_matrix + epsilon @ self.residual_cholesky.T
        value_matrix = torch.exp(ln_value_matrix)

        means: TensorMap = {}
        ln_values: TensorMap = {}
        values: TensorMap = {}
        for index, name in enumerate(self.output_names):
            means[name] = mean_matrix[:, index]
            ln_values[name] = ln_value_matrix[:, index]
            values[name] = value_matrix[:, index]

        return GroundMotionSequence(
            values=values,
            ln_values=ln_values,
            means=means,
            residuals={name: residuals[name] for name in self.residual_names},
            auxiliary={"features": features},
        )

    def log_prob(self, ground_motion: GroundMotionSequence) -> Tensor:
        if not self.residual_names:
            return torch.zeros(())
        residual_matrix = torch.stack(
            [ground_motion.residuals[name] for name in self.residual_names], dim=-1
        )
        return (
            -0.5
            * (
                math.log(2.0 * math.pi)
                + residual_matrix.square()
            )
        ).sum()
