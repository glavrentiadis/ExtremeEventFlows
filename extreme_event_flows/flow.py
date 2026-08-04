from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import Tensor, nn

from .application import GroundMotionApplication
from .containers import FlowSequenceSample, SimulationSample, TensorMap
from .transforms import IdentityTransform, ScalarTransform


def normal_logpdf(x: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    standardized = (x - mean) * torch.exp(-log_std)
    return (
        -0.5 * standardized.square()
        - log_std
        - 0.5 * math.log(2.0 * math.pi)
    )


def normal_log_survival(threshold: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    standardized = (threshold - mean) * torch.exp(-log_std)
    return torch.special.log_ndtr(-standardized)


class ConditionalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class AutoregressiveEventFlow(nn.Module):
    r"""
    Event-by-event normalizing flow with an arbitrary event dimension.

    The variable order is

        [time increment, remaining source variables, GMM residual variables].

    Within each event, variable j is conditioned on the sequence history and
    all preceding raw variables in the same event. This triangular structure
    defines a tractable joint event density for any number of source parameters
    and any number of ground-motion residuals.
    """

    def __init__(
        self,
        *,
        application: GroundMotionApplication,
        hidden_dim: int = 128,
        network_dim: int = 128,
        minimum_log_std: float = -5.0,
        maximum_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        if maximum_log_std <= minimum_log_std:
            raise ValueError("maximum_log_std must exceed minimum_log_std.")

        # Keep a non-registered reference. The physical application is managed
        # and moved to device independently from the trainable proposal.
        object.__setattr__(self, "application", application)
        self.hidden_dim = int(hidden_dim)
        self.minimum_log_std = float(minimum_log_std)
        self.maximum_log_std = float(maximum_log_std)
        self.identity_transform = IdentityTransform()

        source_names = list(application.source_model.event_names)
        time_name = application.source_model.time_increment_name
        source_names.remove(time_name)
        self.source_names = (time_name, *source_names)
        self.residual_names = application.ground_motion_model.residual_names
        self.variable_names = (*self.source_names, *self.residual_names)
        self.time_increment_name = time_name
        self.event_dimension = len(self.variable_names)

        context_dim = hidden_dim + 2  # current time and remaining time
        self.parameter_networks = nn.ModuleList(
            ConditionalMLP(context_dim + index, network_dim, 2)
            for index in range(self.event_dimension)
        )
        self.recurrent_cell = nn.GRUCell(
            input_size=self.event_dimension + 1,
            hidden_size=hidden_dim,
        )
        self.initial_hidden = nn.Parameter(torch.zeros(1, hidden_dim))

    def _transform(self, variable_name: str) -> ScalarTransform:
        if variable_name in self.source_names:
            return self.application.source_model.event_transforms[variable_name]
        return self.identity_transform

    def _bounded_log_std(self, raw_value: Tensor) -> Tensor:
        fraction = torch.sigmoid(raw_value)
        return self.minimum_log_std + (
            self.maximum_log_std - self.minimum_log_std
        ) * fraction

    def _context(
        self,
        hidden: Tensor,
        current_time: Tensor,
        exposure_time: Tensor,
    ) -> Tensor:
        remaining = torch.clamp(exposure_time - current_time, min=0.0)
        time_features = torch.stack(
            [current_time / exposure_time, remaining / exposure_time]
        ).reshape(1, 2)
        return torch.cat([hidden, time_features], dim=-1)

    def _distribution_parameters(
        self,
        variable_index: int,
        hidden: Tensor,
        current_time: Tensor,
        exposure_time: Tensor,
        raw_prefix: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        context = self._context(hidden, current_time, exposure_time)
        if raw_prefix:
            prefix = torch.stack(raw_prefix).reshape(1, -1)
            network_input = torch.cat([context, prefix], dim=-1)
        else:
            network_input = context
        parameters = self.parameter_networks[variable_index](network_input)[0]
        return parameters[0], self._bounded_log_std(parameters[1])

    def _update_hidden(
        self,
        hidden: Tensor,
        raw_event: Tensor,
        event_time: Tensor,
        exposure_time: Tensor,
    ) -> Tensor:
        event_input = torch.cat(
            [raw_event, (event_time / exposure_time).reshape(1)]
        ).reshape(1, -1)
        return self.recurrent_cell(event_input, hidden)

    def _log_stop_probability(
        self,
        hidden: Tensor,
        current_time: Tensor,
        exposure_time: Tensor,
    ) -> Tensor:
        remaining = exposure_time - current_time
        transform = self._transform(self.time_increment_name)
        if bool((remaining <= transform.lower_bound).item()):
            return torch.zeros_like(current_time)
        mean, log_std = self._distribution_parameters(
            0, hidden, current_time, exposure_time, []
        )
        raw_threshold = transform.inverse(remaining)
        return normal_log_survival(raw_threshold, mean, log_std)

    def sample_one(
        self,
        *,
        exposure_time: float,
        maximum_events: int = 100_000,
    ) -> FlowSequenceSample:
        if exposure_time <= 0.0:
            raise ValueError("exposure_time must be positive.")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        exposure = torch.as_tensor(exposure_time, device=device, dtype=dtype)
        current_time = torch.zeros((), device=device, dtype=dtype)
        hidden = self.initial_hidden

        collected_source: dict[str, list[Tensor]] = {
            name: [] for name in self.source_names
        }
        collected_residuals: dict[str, list[Tensor]] = {
            name: [] for name in self.residual_names
        }
        raw_events: list[Tensor] = []
        base_samples: list[Tensor] = []
        log_q_events: list[Tensor] = []
        log_q_stop: Tensor | None = None

        for _ in range(maximum_events):
            raw_values: list[Tensor] = []
            base_values: list[Tensor] = []
            physical_values: dict[str, Tensor] = {}
            log_q_raw = torch.zeros((), device=device, dtype=dtype)
            log_det = torch.zeros((), device=device, dtype=dtype)

            # Generate the time increment first, permitting termination before
            # the remaining source and ground-motion variables are generated.
            mean, log_std = self._distribution_parameters(
                0, hidden, current_time, exposure, raw_values
            )
            z = torch.randn((), device=device, dtype=dtype)
            raw_dt = mean + torch.exp(log_std) * z
            dt_transform = self._transform(self.time_increment_name)
            dt = dt_transform(raw_dt)
            proposed_time = current_time + dt

            if bool((proposed_time > exposure).item()):
                log_q_stop = self._log_stop_probability(
                    hidden, current_time, exposure
                )
                break

            raw_values.append(raw_dt)
            base_values.append(z)
            physical_values[self.time_increment_name] = dt
            log_q_raw = log_q_raw + normal_logpdf(raw_dt, mean, log_std)
            log_det = log_det + dt_transform.log_abs_det_jacobian(raw_dt)

            for variable_index, variable_name in enumerate(
                self.variable_names[1:], start=1
            ):
                mean, log_std = self._distribution_parameters(
                    variable_index,
                    hidden,
                    current_time,
                    exposure,
                    raw_values,
                )
                z = torch.randn((), device=device, dtype=dtype)
                raw = mean + torch.exp(log_std) * z
                transform = self._transform(variable_name)
                physical = transform(raw)

                raw_values.append(raw)
                base_values.append(z)
                physical_values[variable_name] = physical
                log_q_raw = log_q_raw + normal_logpdf(raw, mean, log_std)
                log_det = log_det + transform.log_abs_det_jacobian(raw)

            for name in self.source_names:
                collected_source[name].append(physical_values[name])
            for name in self.residual_names:
                collected_residuals[name].append(physical_values[name])

            raw_event = torch.stack(raw_values)
            raw_events.append(raw_event)
            base_samples.append(torch.stack(base_values))
            log_q_events.append(log_q_raw - log_det)
            hidden = self._update_hidden(hidden, raw_event, proposed_time, exposure)
            current_time = proposed_time
        else:
            raise RuntimeError("maximum_events reached before sequence termination.")

        if log_q_stop is None:
            raise RuntimeError("The stopping probability was not evaluated.")

        source_values: TensorMap = {}
        residual_values: TensorMap = {}
        for name, values in collected_source.items():
            source_values[name] = (
                torch.stack(values)
                if values
                else torch.empty(0, device=device, dtype=dtype)
            )
        for name, values in collected_residuals.items():
            residual_values[name] = (
                torch.stack(values)
                if values
                else torch.empty(0, device=device, dtype=dtype)
            )

        sample = self.application.build_sample(
            source_values=source_values,
            residual_values=residual_values,
        )
        raw_event_tensor = (
            torch.stack(raw_events)
            if raw_events
            else torch.empty((0, self.event_dimension), device=device, dtype=dtype)
        )
        base_tensor = (
            torch.stack(base_samples)
            if base_samples
            else torch.empty((0, self.event_dimension), device=device, dtype=dtype)
        )
        log_event_tensor = (
            torch.stack(log_q_events)
            if log_q_events
            else torch.empty(0, device=device, dtype=dtype)
        )
        return FlowSequenceSample(
            sample=sample,
            raw_events=raw_event_tensor,
            base_samples=base_tensor,
            log_q_events=log_event_tensor,
            log_q_stop=log_q_stop,
            log_q=log_event_tensor.sum() + log_q_stop,
        )

    def _raw_event_from_sample(
        self,
        sample: SimulationSample,
        event_index: int,
    ) -> Tensor:
        raw_values: list[Tensor] = []
        for name in self.source_names:
            raw_values.append(
                self._transform(name).inverse(sample.source[name][event_index])
            )
        for name in self.residual_names:
            raw_values.append(sample.ground_motion.residuals[name][event_index])
        return torch.stack(raw_values)

    def log_prob_one(
        self,
        sample: SimulationSample,
        *,
        exposure_time: float,
    ) -> Tensor:
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        exposure = torch.as_tensor(exposure_time, device=device, dtype=dtype)
        current_time = torch.zeros((), device=device, dtype=dtype)
        hidden = self.initial_hidden
        log_q = torch.zeros((), device=device, dtype=dtype)

        for event_index in range(sample.source.number_of_events):
            raw_event = self._raw_event_from_sample(sample, event_index)
            raw_prefix: list[Tensor] = []
            log_q_raw = torch.zeros_like(log_q)
            log_det = torch.zeros_like(log_q)

            for variable_index, variable_name in enumerate(self.variable_names):
                raw = raw_event[variable_index]
                mean, log_std = self._distribution_parameters(
                    variable_index,
                    hidden,
                    current_time,
                    exposure,
                    raw_prefix,
                )
                transform = self._transform(variable_name)
                log_q_raw = log_q_raw + normal_logpdf(raw, mean, log_std)
                log_det = log_det + transform.log_abs_det_jacobian(raw)
                raw_prefix.append(raw)

            event_time = sample.source.event_time[event_index]
            expected_time = current_time + sample.source.dt[event_index]
            if not torch.allclose(
                event_time, expected_time, rtol=1.0e-6, atol=1.0e-8
            ):
                raise ValueError("event_time is inconsistent with the source dt.")
            if bool((event_time > exposure).item()):
                return torch.full_like(log_q, -torch.inf)

            log_q = log_q + log_q_raw - log_det
            hidden = self._update_hidden(hidden, raw_event, event_time, exposure)
            current_time = event_time

        return log_q + self._log_stop_probability(
            hidden, current_time, exposure
        )
