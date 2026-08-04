from __future__ import annotations

import math

import torch
from torch import Tensor

from .flow import AutoregressiveEventFlow
from .target import RareEventTargetDensity


def _validate_sample_count(number_of_samples: int) -> None:
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be positive.")


def _normal_critical_value(
    confidence_level: float,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tensor:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")

    probability = 0.5 * (1.0 + confidence_level)
    standard_normal = torch.distributions.Normal(
        torch.zeros((), device=device, dtype=dtype),
        torch.ones((), device=device, dtype=dtype),
    )
    return standard_normal.icdf(
        torch.tensor(probability, device=device, dtype=dtype)
    )


def _relative_standard_error(probability: Tensor, standard_error: Tensor) -> Tensor:
    return torch.where(
        probability > 0.0,
        standard_error / probability,
        torch.full_like(probability, torch.inf),
    )


def _normal_confidence_interval(
    probability: Tensor,
    standard_error: Tensor,
    *,
    confidence_level: float,
) -> tuple[Tensor, Tensor]:
    critical_value = _normal_critical_value(
        confidence_level,
        device=probability.device,
        dtype=probability.dtype,
    )
    lower = torch.clamp(probability - critical_value * standard_error, 0.0, 1.0)
    upper = torch.clamp(probability + critical_value * standard_error, 0.0, 1.0)
    return lower, upper


def _wilson_confidence_interval(
    probability: Tensor,
    number_of_samples: int,
    *,
    confidence_level: float,
) -> tuple[Tensor, Tensor]:
    """Wilson interval for a Bernoulli probability."""
    critical_value = _normal_critical_value(
        confidence_level,
        device=probability.device,
        dtype=probability.dtype,
    )
    n = torch.tensor(
        float(number_of_samples),
        device=probability.device,
        dtype=probability.dtype,
    )
    z_squared = critical_value.square()
    denominator = 1.0 + z_squared / n
    center = (probability + z_squared / (2.0 * n)) / denominator
    half_width = (
        critical_value
        * torch.sqrt(
            probability * (1.0 - probability) / n
            + z_squared / (4.0 * n.square())
        )
        / denominator
    )
    return (
        torch.clamp(center - half_width, 0.0, 1.0),
        torch.clamp(center + half_width, 0.0, 1.0),
    )


@torch.no_grad()
def estimate_probability_flow_only(
    flow: AutoregressiveEventFlow,
    target: RareEventTargetDensity,
    *,
    number_of_samples: int,
    confidence_level: float = 0.95,
) -> dict[str, Tensor]:
    """Importance-sampling estimate using only the autoregressive flow.

    This estimator is useful as a diagnostic. For final estimation, a
    defensive mixture is usually safer because it guarantees support from the
    original physical model.
    """
    _validate_sample_count(number_of_samples)

    log_weights: list[Tensor] = []
    contributions: list[Tensor] = []
    indicators: list[Tensor] = []

    for _ in range(number_of_samples):
        proposal = flow.sample_one(exposure_time=target.exposure_time)
        evaluation = target(proposal.sample)
        log_weight = evaluation.log_original_density - proposal.log_q
        weight = torch.exp(log_weight)
        indicator = evaluation.rare_event_indicator.to(weight.dtype)

        log_weights.append(log_weight)
        contributions.append(indicator * weight)
        indicators.append(evaluation.rare_event_indicator)

    stacked_weights = torch.stack(log_weights)
    contribution_tensor = torch.stack(contributions)
    indicator_tensor = torch.stack(indicators).to(stacked_weights.dtype)

    probability = contribution_tensor.mean()
    if number_of_samples > 1:
        contribution_variance = contribution_tensor.var(unbiased=True)
        log_weight_std = stacked_weights.std(unbiased=True)
    else:
        contribution_variance = torch.zeros_like(probability)
        log_weight_std = torch.zeros_like(probability)

    estimator_variance = contribution_variance / float(number_of_samples)
    standard_error = torch.sqrt(estimator_variance)
    ci_lower, ci_upper = _normal_confidence_interval(
        probability,
        standard_error,
        confidence_level=confidence_level,
    )

    centered = torch.exp(stacked_weights - stacked_weights.max())
    ess = centered.sum().square() / (centered.square().sum() + 1.0e-12)

    return {
        "probability": probability,
        "standard_error": standard_error,
        "relative_standard_error": _relative_standard_error(
            probability, standard_error
        ),
        "confidence_interval_lower": ci_lower,
        "confidence_interval_upper": ci_upper,
        "confidence_level": torch.tensor(
            confidence_level,
            device=probability.device,
            dtype=probability.dtype,
        ),
        "sample_variance": contribution_variance,
        "estimator_variance": estimator_variance,
        "number_of_samples": torch.tensor(
            number_of_samples,
            device=probability.device,
            dtype=torch.long,
        ),
        "exceedance_count": indicator_tensor.sum().to(torch.long),
        "indicator_rate_under_q": indicator_tensor.mean(),
        "mean_weight": torch.exp(
            torch.logsumexp(stacked_weights, dim=0) - math.log(number_of_samples)
        ),
        "effective_sample_size": ess,
        "log_weight_mean": stacked_weights.mean(),
        "log_weight_std": log_weight_std,
    }


@torch.no_grad()
def estimate_probability_defensive_mixture(
    flow: AutoregressiveEventFlow,
    target: RareEventTargetDensity,
    *,
    number_of_samples: int,
    original_fraction: float,
    device: torch.device | str,
    dtype: torch.dtype,
    confidence_level: float = 0.95,
) -> dict[str, Tensor]:
    r"""Estimate failure probability with a defensive mixture proposal.

    The sampling density is

        r(x) = (1 - epsilon) q_theta(x) + epsilon p(x),

    and the estimator is

        mean[ I_F(x) p(x) / r(x) ].

    Parameters
    ----------
    original_fraction
        Defensive-mixture weight ``epsilon``. It must lie strictly between
        zero and one.
    """
    _validate_sample_count(number_of_samples)
    if not 0.0 < original_fraction < 1.0:
        raise ValueError("original_fraction must lie strictly between zero and one.")

    log_flow_fraction = torch.log(
        torch.tensor(1.0 - original_fraction, device=device, dtype=dtype)
    )
    log_original_fraction = torch.log(
        torch.tensor(original_fraction, device=device, dtype=dtype)
    )

    log_weights: list[Tensor] = []
    contributions: list[Tensor] = []
    indicators: list[Tensor] = []
    original_draws = 0

    for _ in range(number_of_samples):
        draw_from_original = bool(
            (torch.rand((), device=device) < original_fraction).item()
        )
        if draw_from_original:
            original_draws += 1
            sample = target.application.sample_original(
                target.exposure_time,
                device=device,
                dtype=dtype,
            )
        else:
            sample = flow.sample_one(exposure_time=target.exposure_time).sample

        evaluation = target(sample)
        log_p = evaluation.log_original_density
        log_q = flow.log_prob_one(sample, exposure_time=target.exposure_time)
        log_r = torch.logaddexp(
            log_flow_fraction + log_q,
            log_original_fraction + log_p,
        )
        log_weight = log_p - log_r
        weight = torch.exp(log_weight)
        indicator = evaluation.rare_event_indicator.to(dtype)

        log_weights.append(log_weight)
        contributions.append(indicator * weight)
        indicators.append(evaluation.rare_event_indicator)

    stacked_weights = torch.stack(log_weights)
    contribution_tensor = torch.stack(contributions)
    indicator_tensor = torch.stack(indicators).to(dtype)

    probability = contribution_tensor.mean()
    if number_of_samples > 1:
        contribution_variance = contribution_tensor.var(unbiased=True)
        log_weight_std = stacked_weights.std(unbiased=True)
    else:
        contribution_variance = torch.zeros_like(probability)
        log_weight_std = torch.zeros_like(probability)

    estimator_variance = contribution_variance / float(number_of_samples)
    standard_error = torch.sqrt(estimator_variance)
    ci_lower, ci_upper = _normal_confidence_interval(
        probability,
        standard_error,
        confidence_level=confidence_level,
    )

    centered = torch.exp(stacked_weights - stacked_weights.max())
    ess = centered.sum().square() / (centered.square().sum() + 1.0e-12)

    return {
        "probability": probability,
        "standard_error": standard_error,
        "relative_standard_error": _relative_standard_error(
            probability, standard_error
        ),
        "confidence_interval_lower": ci_lower,
        "confidence_interval_upper": ci_upper,
        "confidence_level": torch.tensor(
            confidence_level,
            device=probability.device,
            dtype=probability.dtype,
        ),
        "sample_variance": contribution_variance,
        "estimator_variance": estimator_variance,
        "number_of_samples": torch.tensor(
            number_of_samples,
            device=probability.device,
            dtype=torch.long,
        ),
        "exceedance_count": indicator_tensor.sum().to(torch.long),
        "indicator_rate_under_mixture": indicator_tensor.mean(),
        "original_draw_count": torch.tensor(
            original_draws,
            device=probability.device,
            dtype=torch.long,
        ),
        "flow_draw_count": torch.tensor(
            number_of_samples - original_draws,
            device=probability.device,
            dtype=torch.long,
        ),
        "mean_weight": torch.exp(
            torch.logsumexp(stacked_weights, dim=0) - math.log(number_of_samples)
        ),
        "effective_sample_size": ess,
        "log_weight_mean": stacked_weights.mean(),
        "log_weight_std": log_weight_std,
    }


@torch.no_grad()
def estimate_probability_naive_monte_carlo(
    target: RareEventTargetDensity,
    *,
    number_of_samples: int,
    device: torch.device | str,
    dtype: torch.dtype,
    confidence_level: float = 0.95,
) -> dict[str, Tensor]:
    r"""Estimate the rare-event probability by naïve Monte Carlo.

    Independent samples are drawn directly from the original physical model,

        X_j ~ p(x),

    and the estimator is the sample mean of the failure indicator,

        P_hat = (1 / N) sum_j I_F(X_j).

    The returned confidence interval is the Wilson binomial interval, which is
    more informative than the symmetric normal interval when few or no
    exceedances are observed.
    """
    _validate_sample_count(number_of_samples)

    indicators: list[Tensor] = []
    performances: list[Tensor] = []

    for _ in range(number_of_samples):
        sample = target.application.sample_original(
            target.exposure_time,
            device=device,
            dtype=dtype,
        )
        evaluation = target(sample)
        indicators.append(evaluation.rare_event_indicator)
        performances.append(evaluation.performance)

    indicator_tensor = torch.stack(indicators).to(dtype)
    performance_tensor = torch.stack(performances)

    probability = indicator_tensor.mean()
    if number_of_samples > 1:
        sample_variance = indicator_tensor.var(unbiased=True)
    else:
        sample_variance = torch.zeros_like(probability)
    estimator_variance = sample_variance / float(number_of_samples)
    standard_error = torch.sqrt(estimator_variance)
    ci_lower, ci_upper = _wilson_confidence_interval(
        probability,
        number_of_samples,
        confidence_level=confidence_level,
    )

    finite_performance = performance_tensor[torch.isfinite(performance_tensor)]
    if finite_performance.numel() > 0:
        performance_mean = finite_performance.mean()
        if finite_performance.numel() > 1:
            performance_std = finite_performance.std(unbiased=True)
        else:
            performance_std = torch.zeros_like(probability)
    else:
        performance_mean = torch.full_like(probability, torch.nan)
        performance_std = torch.full_like(probability, torch.nan)

    zero_event_count = (~torch.isfinite(performance_tensor)).sum().to(torch.long)

    return {
        "probability": probability,
        "standard_error": standard_error,
        "relative_standard_error": _relative_standard_error(
            probability, standard_error
        ),
        "confidence_interval_lower": ci_lower,
        "confidence_interval_upper": ci_upper,
        "confidence_level": torch.tensor(
            confidence_level,
            device=probability.device,
            dtype=probability.dtype,
        ),
        "sample_variance": sample_variance,
        "estimator_variance": estimator_variance,
        "number_of_samples": torch.tensor(
            number_of_samples,
            device=probability.device,
            dtype=torch.long,
        ),
        "exceedance_count": indicator_tensor.sum().to(torch.long),
        "performance_mean": performance_mean,
        "performance_std": performance_std,
        "finite_performance_count": torch.tensor(
            finite_performance.numel(),
            device=probability.device,
            dtype=torch.long,
        ),
        "zero_event_count": zero_event_count,
        "zero_event_fraction": zero_event_count.to(dtype) / float(number_of_samples),
    }


@torch.no_grad()
def estimate_probability_crude(
    target: RareEventTargetDensity,
    *,
    number_of_samples: int,
    device: torch.device | str,
    dtype: torch.dtype,
    confidence_level: float = 0.95,
) -> dict[str, Tensor]:
    """Backward-compatible alias for naïve Monte Carlo estimation."""
    return estimate_probability_naive_monte_carlo(
        target,
        number_of_samples=number_of_samples,
        device=device,
        dtype=dtype,
        confidence_level=confidence_level,
    )


def cross_entropy_update(
    flow: AutoregressiveEventFlow,
    target: RareEventTargetDensity,
    optimizer: torch.optim.Optimizer,
    *,
    number_of_samples: int,
    temperature: float = 1.0,
) -> dict[str, Tensor]:
    """Perform one weighted maximum-likelihood cross-entropy update."""
    _validate_sample_count(number_of_samples)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    samples = []
    log_importance_scores = []
    with torch.no_grad():
        for _ in range(number_of_samples):
            proposal = flow.sample_one(exposure_time=target.exposure_time)
            evaluation = target(proposal.sample)
            samples.append(proposal.sample)
            log_importance_scores.append(
                (evaluation.log_target_unnormalized - proposal.log_q) / temperature
            )
        normalized_weights = torch.softmax(torch.stack(log_importance_scores), dim=0)

    log_probabilities = torch.stack(
        [
            flow.log_prob_one(sample, exposure_time=target.exposure_time)
            for sample in samples
        ]
    )
    loss = -(normalized_weights * log_probabilities).sum()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
    optimizer.step()

    return {
        "loss": loss.detach(),
        "maximum_normalized_weight": normalized_weights.max(),
        "weight_entropy": -(
            normalized_weights * torch.log(normalized_weights + 1.0e-30)
        ).sum(),
    }
