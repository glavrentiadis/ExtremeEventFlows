from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from .containers import SimulationSample
from .flow import AutoregressiveEventFlow
from .target import RareEventTargetDensity


def select_representative_sequence(
    target: RareEventTargetDensity,
    *,
    method: Literal["naive", "flow"],
    flow: AutoregressiveEventFlow | None = None,
    number_of_candidates: int = 500,
    prefer_exceedance: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[SimulationSample, bool]:
    """Select one nonempty sequence for visualization.

    Candidates are generated from the requested sampler. When
    ``prefer_exceedance`` is true, the first exceeding, nonempty sequence is
    returned. Otherwise, or when no exceedance is found, the nonempty sequence
    with the largest performance value is returned.

    This function is for visualization only; it is not an estimator.
    """
    if number_of_candidates <= 0:
        raise ValueError("number_of_candidates must be positive.")
    if method == "flow" and flow is None:
        raise ValueError("flow is required when method='flow'.")

    best_sample: SimulationSample | None = None
    best_performance = -torch.inf
    best_exceeds = False

    with torch.no_grad():
        for _ in range(number_of_candidates):
            if method == "naive":
                sample = target.application.sample_original(
                    target.exposure_time,
                    device=device,
                    dtype=dtype,
                )
            elif method == "flow":
                assert flow is not None
                sample = flow.sample_one(
                    exposure_time=target.exposure_time
                ).sample
            else:
                raise ValueError("method must be 'naive' or 'flow'.")

            if sample.source.number_of_events == 0:
                continue

            evaluation = target(sample)
            exceeds = bool(evaluation.rare_event_indicator.item())
            performance = float(evaluation.performance.detach().cpu())

            if prefer_exceedance and exceeds:
                return sample, True
            if best_sample is None or performance > best_performance:
                best_sample = sample
                best_performance = performance
                best_exceeds = exceeds

    if best_sample is None:
        raise RuntimeError(
            "No nonempty sequence was generated. Increase number_of_candidates "
            "or the exposure time."
        )
    return best_sample, best_exceeds


def plot_sequence_comparison(
    naive_sample: SimulationSample,
    flow_sample: SimulationSample,
    *,
    ground_motion_name: str,
    threshold: float,
    output_path: str | Path,
    exposure_time: float | None = None,
    title: str | None = None,
    dpi: int = 180,
) -> Path:
    """Plot event magnitude and ground motion versus time for two sequences.

    The first panel compares earthquake magnitude versus event time. The second
    compares the selected ground-motion output versus event time and includes
    the active exceedance threshold.
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be positive.")
    if exposure_time is not None and exposure_time <= 0.0:
        raise ValueError("exposure_time must be positive when provided.")
    if "magnitude" not in naive_sample.source.values:
        raise KeyError("The naive sequence does not contain 'magnitude'.")
    if "magnitude" not in flow_sample.source.values:
        raise KeyError("The flow sequence does not contain 'magnitude'.")
    if ground_motion_name not in naive_sample.ground_motion.values:
        raise KeyError(
            f"Naive sequence does not contain {ground_motion_name!r}."
        )
    if ground_motion_name not in flow_sample.ground_motion.values:
        raise KeyError(
            f"Flow sequence does not contain {ground_motion_name!r}."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_sequence_comparison."
        ) from exc

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    naive_time = naive_sample.source.event_time.detach().cpu().numpy()
    flow_time = flow_sample.source.event_time.detach().cpu().numpy()
    naive_magnitude = naive_sample.source["magnitude"].detach().cpu().numpy()
    flow_magnitude = flow_sample.source["magnitude"].detach().cpu().numpy()
    naive_motion = (
        naive_sample.ground_motion.values[ground_motion_name]
        .detach()
        .cpu()
        .numpy()
    )
    flow_motion = (
        flow_sample.ground_motion.values[ground_motion_name]
        .detach()
        .cpu()
        .numpy()
    )

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.5), sharex=True)

    axes[0].plot(
        naive_time,
        naive_magnitude,
        linestyle="none",
        marker="o",
        label="Naive Monte Carlo",
    )
    axes[0].plot(
        flow_time,
        flow_magnitude,
        linestyle="none",
        marker="^",
        label="Normalizing flow",
    )
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Sampled event magnitudes")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        naive_time,
        naive_motion,
        linestyle="none",
        marker="o",
        label="Naive Monte Carlo",
    )
    axes[1].plot(
        flow_time,
        flow_motion,
        linestyle="none",
        marker="^",
        label="Normalizing flow",
    )
    axes[1].axhline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:g} g",
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(f"{ground_motion_name} (g)")
    axes[1].set_title("Ground motion at the site")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    if exposure_time is not None:
        axes[0].set_xlim(0.0, exposure_time)
        axes[1].set_xlim(0.0, exposure_time)

    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination
