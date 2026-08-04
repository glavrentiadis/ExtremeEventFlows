from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal

from .containers import SourceSequence, TensorMap
from .transforms import ScalarTransform


@dataclass
class MarkovSourceState:
    """State conditioning the next joint source-event distribution."""

    previous_event: TensorMap
    previous_event_time: Tensor
    event_index: Tensor


class JointMarkovSourceKernel(nn.Module, ABC):
    """Joint transition density for all source parameters of one event."""

    @property
    @abstractmethod
    def event_names(self) -> tuple[str, ...]:
        pass

    @property
    @abstractmethod
    def time_increment_name(self) -> str:
        pass

    @property
    @abstractmethod
    def event_transforms(self) -> Mapping[str, ScalarTransform]:
        pass

    @abstractmethod
    def initial_state(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> MarkovSourceState:
        pass

    @abstractmethod
    def sample_event(self, state: MarkovSourceState) -> TensorMap:
        """Sample all source variables jointly for the next event."""

    @abstractmethod
    def log_prob_event(
        self,
        event: Mapping[str, Tensor],
        state: MarkovSourceState,
    ) -> Tensor:
        """Evaluate the joint event log density."""

    @abstractmethod
    def log_survival_time(
        self,
        remaining_time: Tensor,
        state: MarkovSourceState,
    ) -> Tensor:
        """Evaluate the marginal probability that the next dt exceeds a limit."""

    @abstractmethod
    def update_state(
        self,
        state: MarkovSourceState,
        *,
        event: Mapping[str, Tensor],
        event_time: Tensor,
    ) -> MarkovSourceState:
        pass


class GaussianLatentMarkovKernel(JointMarkovSourceKernel):
    r"""
    General joint Markov kernel in a transformed latent Gaussian space.

    Let ``y_i`` contain any number of physical event variables, such as
    ``dt``, ``magnitude``, ``x``, ``y``, ``depth``, stress drop, or mechanism
    parameters. Each coordinate has an invertible scalar transform

        y_{i,j} = T_j(z_{i,j}).

    Conditional on the previous event,

        z_i | y_{i-1} ~ N(mu_i, L L^T),

    with

        mu_i = mu_0 + A ((y_{i-1} - y_ref) / y_scale).

    The covariance creates a fully joint current-event distribution, while the
    transition matrix creates Markov dependence on every selected parameter of
    the preceding event.
    """

    def __init__(
        self,
        *,
        event_transforms: Mapping[str, ScalarTransform],
        time_increment_name: str,
        base_mean: Tensor | Sequence[float],
        transition_matrix: Tensor | Sequence[Sequence[float]],
        latent_cholesky: Tensor | Sequence[Sequence[float]],
        reference_event: Mapping[str, float],
        state_scale: Mapping[str, float],
    ) -> None:
        super().__init__()

        if not event_transforms:
            raise ValueError("event_transforms cannot be empty.")

        self._event_names = tuple(event_transforms)
        if time_increment_name not in self._event_names:
            raise KeyError("time_increment_name must be an event variable.")
        self._time_increment_name = time_increment_name
        self.transforms = nn.ModuleDict(dict(event_transforms))

        dimension = len(self._event_names)
        base_mean_tensor = torch.as_tensor(base_mean, dtype=torch.get_default_dtype())
        transition_tensor = torch.as_tensor(
            transition_matrix, dtype=torch.get_default_dtype()
        )
        cholesky_tensor = torch.as_tensor(
            latent_cholesky, dtype=torch.get_default_dtype()
        )

        if base_mean_tensor.shape != (dimension,):
            raise ValueError(f"base_mean must have shape [{dimension}].")
        if transition_tensor.shape != (dimension, dimension):
            raise ValueError(
                f"transition_matrix must have shape [{dimension}, {dimension}]."
            )
        if cholesky_tensor.shape != (dimension, dimension):
            raise ValueError(
                f"latent_cholesky must have shape [{dimension}, {dimension}]."
            )
        if not torch.allclose(cholesky_tensor, torch.tril(cholesky_tensor)):
            raise ValueError("latent_cholesky must be lower triangular.")
        if torch.any(torch.diagonal(cholesky_tensor) <= 0.0):
            raise ValueError("latent_cholesky must have a positive diagonal.")

        missing_reference = set(self._event_names) - set(reference_event)
        missing_scale = set(self._event_names) - set(state_scale)
        if missing_reference:
            raise KeyError(f"Missing reference values: {sorted(missing_reference)}")
        if missing_scale:
            raise KeyError(f"Missing state scales: {sorted(missing_scale)}")

        reference_vector = torch.tensor(
            [reference_event[name] for name in self._event_names],
            dtype=torch.get_default_dtype(),
        )
        scale_vector = torch.tensor(
            [state_scale[name] for name in self._event_names],
            dtype=torch.get_default_dtype(),
        )
        if torch.any(scale_vector <= 0.0):
            raise ValueError("All state scales must be positive.")

        self.register_buffer("base_mean", base_mean_tensor)
        self.register_buffer("transition_matrix", transition_tensor)
        self.register_buffer("latent_cholesky", cholesky_tensor)
        self.register_buffer("reference_vector", reference_vector)
        self.register_buffer("state_scale_vector", scale_vector)

    @property
    def event_names(self) -> tuple[str, ...]:
        return self._event_names

    @property
    def time_increment_name(self) -> str:
        return self._time_increment_name

    @property
    def event_transforms(self) -> Mapping[str, ScalarTransform]:
        return {name: self.transforms[name] for name in self._event_names}

    def _event_vector(self, event: Mapping[str, Tensor]) -> Tensor:
        missing = set(self._event_names) - set(event)
        if missing:
            raise KeyError(f"Missing event variables: {sorted(missing)}")
        return torch.stack([event[name] for name in self._event_names])

    def _reference_event(self) -> TensorMap:
        return {
            name: self.reference_vector[index]
            for index, name in enumerate(self._event_names)
        }

    def _conditional_mean(self, state: MarkovSourceState) -> Tensor:
        previous = self._event_vector(state.previous_event)
        feature = (previous - self.reference_vector) / self.state_scale_vector
        return self.base_mean + self.transition_matrix @ feature

    def _latent_distribution(self, state: MarkovSourceState) -> MultivariateNormal:
        return MultivariateNormal(
            loc=self._conditional_mean(state),
            scale_tril=self.latent_cholesky,
        )

    def initial_state(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> MarkovSourceState:
        previous = {
            name: value.to(device=device, dtype=dtype)
            for name, value in self._reference_event().items()
        }
        return MarkovSourceState(
            previous_event=previous,
            previous_event_time=torch.zeros((), device=device, dtype=dtype),
            event_index=torch.zeros((), device=device, dtype=torch.long),
        )

    def sample_event(self, state: MarkovSourceState) -> TensorMap:
        raw = self._latent_distribution(state).rsample()
        return {
            name: self.transforms[name](raw[index])
            for index, name in enumerate(self._event_names)
        }

    def log_prob_event(
        self,
        event: Mapping[str, Tensor],
        state: MarkovSourceState,
    ) -> Tensor:
        raw_values: list[Tensor] = []
        log_det = torch.zeros_like(next(iter(event.values())))

        for name in self._event_names:
            physical = event[name]
            raw = self.transforms[name].inverse(physical)
            raw_values.append(raw)
            log_det = log_det + self.transforms[name].log_abs_det_jacobian(raw)

        raw_vector = torch.stack(raw_values)
        return self._latent_distribution(state).log_prob(raw_vector) - log_det

    def log_survival_time(
        self,
        remaining_time: Tensor,
        state: MarkovSourceState,
    ) -> Tensor:
        transform = self.transforms[self.time_increment_name]
        if bool((remaining_time <= transform.lower_bound).item()):
            return torch.zeros_like(remaining_time)

        time_index = self._event_names.index(self.time_increment_name)
        mean = self._conditional_mean(state)[time_index]
        variance = self.latent_cholesky[time_index].square().sum()
        std = torch.sqrt(variance)
        raw_threshold = transform.inverse(remaining_time)
        standardized = (raw_threshold - mean) / std
        return torch.special.log_ndtr(-standardized)

    def update_state(
        self,
        state: MarkovSourceState,
        *,
        event: Mapping[str, Tensor],
        event_time: Tensor,
    ) -> MarkovSourceState:
        return MarkovSourceState(
            previous_event={name: event[name] for name in self._event_names},
            previous_event_time=event_time,
            event_index=state.event_index + 1,
        )


class MarkovJointSourceModel(nn.Module):
    """Variable-length source process built from a general joint Markov kernel."""

    def __init__(self, *, name: str, joint_kernel: JointMarkovSourceKernel) -> None:
        super().__init__()
        self.name = name
        self.joint_kernel = joint_kernel

    @property
    def event_names(self) -> tuple[str, ...]:
        return self.joint_kernel.event_names

    @property
    def time_increment_name(self) -> str:
        return self.joint_kernel.time_increment_name

    @property
    def event_transforms(self) -> Mapping[str, ScalarTransform]:
        return self.joint_kernel.event_transforms

    def make_sequence(self, values: Mapping[str, Tensor]) -> SourceSequence:
        missing = set(self.event_names) - set(values)
        if missing:
            raise KeyError(f"Missing source variables: {sorted(missing)}")
        filtered = {name: values[name] for name in self.event_names}
        dt = filtered[self.time_increment_name]
        event_time = torch.cumsum(dt, dim=0)
        return SourceSequence(
            values=filtered,
            event_time=event_time,
            time_increment_name=self.time_increment_name,
        )

    @torch.no_grad()
    def sample_sequence(
        self,
        exposure_time: float,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        maximum_events: int = 100_000,
    ) -> SourceSequence:
        if exposure_time <= 0.0:
            raise ValueError("exposure_time must be positive.")

        exposure = torch.as_tensor(exposure_time, device=device, dtype=dtype)
        current_time = torch.zeros((), device=device, dtype=dtype)
        state = self.joint_kernel.initial_state(device=device, dtype=dtype)
        collected: dict[str, list[Tensor]] = {name: [] for name in self.event_names}

        for _ in range(maximum_events):
            event = self.joint_kernel.sample_event(state)
            dt = event[self.time_increment_name]
            proposed_time = current_time + dt
            if bool((proposed_time > exposure).item()):
                break

            for name in self.event_names:
                collected[name].append(event[name])

            state = self.joint_kernel.update_state(
                state,
                event=event,
                event_time=proposed_time,
            )
            current_time = proposed_time
        else:
            raise RuntimeError("maximum_events reached before sequence termination.")

        values: TensorMap = {}
        for name in self.event_names:
            if collected[name]:
                values[name] = torch.stack(collected[name])
            else:
                values[name] = torch.empty(0, device=device, dtype=dtype)
        return self.make_sequence(values)

    def log_prob(self, source: SourceSequence, exposure_time: float) -> Tensor:
        dtype = source.event_time.dtype
        device = source.event_time.device
        exposure = torch.as_tensor(exposure_time, device=device, dtype=dtype)
        current_time = torch.zeros((), device=device, dtype=dtype)
        state = self.joint_kernel.initial_state(device=device, dtype=dtype)
        log_density = torch.zeros((), device=device, dtype=dtype)

        for event_index in range(source.number_of_events):
            event = {name: source[name][event_index] for name in self.event_names}
            event_time = source.event_time[event_index]
            expected_time = current_time + event[self.time_increment_name]

            if not torch.allclose(
                event_time, expected_time, rtol=1.0e-6, atol=1.0e-8
            ):
                raise ValueError("event_time is inconsistent with the time increments.")
            if bool((event_time > exposure).item()):
                return torch.full_like(log_density, -torch.inf)

            log_density = log_density + self.joint_kernel.log_prob_event(event, state)
            state = self.joint_kernel.update_state(
                state,
                event=event,
                event_time=event_time,
            )
            current_time = event_time

        remaining_time = torch.clamp(exposure - current_time, min=0.0)
        return log_density + self.joint_kernel.log_survival_time(
            remaining_time,
            state,
        )
