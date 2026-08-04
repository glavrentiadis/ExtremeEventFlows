#!/usr/bin/env python3
"""Continue one flow across progressively higher SA(0.2 s) thresholds.

The flow is pretrained once on the original physical model. It is then adapted
successively to every threshold supplied through ``--thresholds``. The flow
parameters and Adam optimizer state are retained between thresholds.

At each threshold, the script compares defensive-mixture importance sampling
with naïve Monte Carlo. A resumable checkpoint is written after every threshold.
Finally, the script saves a two-panel figure comparing one physical-model
sequence and one normalizing-flow sequence:

1. event magnitude versus event time; and
2. SA(0.2 s) versus event time.

Run from the repository root without installing the package::

    python examples/progressive_threshold_sa_0p2.py \
        --thresholds 0.5 0.7 0.9 1.1

Continue later from the saved flow and optimizer state::

    python examples/progressive_threshold_sa_0p2.py \
        --resume progressive_sa_0p2_checkpoint.pt \
        --thresholds 1.3 1.5

Fast smoke run::

    python examples/progressive_threshold_sa_0p2.py \
        --thresholds 0.4 0.5 \
        --pretrain-updates 1 \
        --pretrain-samples 8 \
        --updates-per-threshold 1 \
        --samples-per-update 8 \
        --importance-samples 20 \
        --naive-samples 20 \
        --plot-candidates 20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extreme_event_flows import (
    AutoregressiveEventFlow,
    BoundedSigmoidTransform,
    GaussianLatentMarkovKernel,
    GroundMotionApplication,
    JointGroundMotionThresholdPerformance,
    MarkovJointSourceModel,
    MultiOutputGaussianGroundMotionModel,
    PenaltyFunction,
    PointSourceFeatureBuilder,
    PositiveSoftplusTransform,
    ProgressiveThresholdTrainer,
    RareEventTargetDensity,
    Site,
    estimate_probability_defensive_mixture,
    estimate_probability_naive_monte_carlo,
    plot_sequence_comparison,
    select_representative_sequence,
)


def build_source_model(*, dtype: torch.dtype) -> MarkovJointSourceModel:
    """Construct the illustrative joint Markov source model."""
    event_transforms = {
        "dt": PositiveSoftplusTransform(minimum=0.0),
        "magnitude": BoundedSigmoidTransform(low=4.0, high=8.5),
        "x": BoundedSigmoidTransform(low=-20.0, high=20.0),
        "y": BoundedSigmoidTransform(low=-20.0, high=20.0),
        "depth": BoundedSigmoidTransform(low=5.0, high=15.0),
    }

    reference_event = {
        "dt": 0.60,
        "magnitude": 5.0,
        "x": 0.0,
        "y": 0.0,
        "depth": 10.0,
    }

    base_mean = torch.stack(
        [
            event_transforms[name].inverse(
                torch.tensor(reference_event[name], dtype=dtype)
            )
            for name in event_transforms
        ]
    )

    transition_matrix = torch.tensor(
        [
            [0.10, -0.08, 0.00, 0.00, 0.00],
            [0.00, 0.15, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.20, 0.02, 0.00],
            [0.00, 0.00, 0.02, 0.20, 0.00],
            [0.00, 0.05, 0.00, 0.00, 0.10],
        ],
        dtype=dtype,
    )

    latent_cholesky = torch.tensor(
        [
            [0.45, 0.00, 0.00, 0.00, 0.00],
            [-0.12, 0.65, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.25, 0.00, 0.00],
            [0.00, 0.00, 0.05, 0.25, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.20],
        ],
        dtype=dtype,
    )

    kernel = GaussianLatentMarkovKernel(
        event_transforms=event_transforms,
        time_increment_name="dt",
        base_mean=base_mean,
        transition_matrix=transition_matrix,
        latent_cholesky=latent_cholesky,
        reference_event=reference_event,
        state_scale={
            "dt": 0.60,
            "magnitude": 1.0,
            "x": 10.0,
            "y": 10.0,
            "depth": 5.0,
        },
    )

    return MarkovJointSourceModel(
        name="joint_markov_point_source",
        joint_kernel=kernel,
    )


def build_problem(
    *,
    device: torch.device,
    dtype: torch.dtype,
    exposure_time: float,
    initial_threshold: float,
    penalty_alpha: float,
) -> tuple[RareEventTargetDensity, AutoregressiveEventFlow]:
    """Build the physical problem and trainable proposal."""
    source_model = build_source_model(dtype=dtype)

    site = Site(
        name="site_A",
        fields={
            "x": 20.0,
            "y": 0.0,
            "depth": 0.0,
            "vs30": 500.0,
        },
    )

    feature_builder = PointSourceFeatureBuilder(
        magnitude_name="magnitude",
        source_location_names=("x", "y", "depth"),
        site_location_names=("x", "y", "depth"),
        vs30_name="vs30",
        distance_offset=5.0,
        reference_vs30=500.0,
    )

    ground_motion_model = MultiOutputGaussianGroundMotionModel(
        feature_builder=feature_builder,
        output_names=("SA_0p2",),
        coefficients=torch.tensor(
            [[-4.00, 1.10, -1.10, -0.20]],
            dtype=dtype,
        ),
        residual_cholesky=torch.tensor(
            [[0.60]],
            dtype=dtype,
        ),
    )

    application = GroundMotionApplication(
        source_model=source_model,
        site=site,
        ground_motion_model=ground_motion_model,
    ).to(device=device, dtype=dtype)

    performance = JointGroundMotionThresholdPerformance(
        thresholds={"SA_0p2": initial_threshold},
        mode="all",
    )

    target = RareEventTargetDensity(
        application=application,
        performance_function=performance,
        penalty_function=PenaltyFunction(
            threshold=0.0,
            alpha=penalty_alpha,
        ),
        exposure_time=exposure_time,
    ).to(device=device, dtype=dtype)

    flow = AutoregressiveEventFlow(
        application=application,
        hidden_dim=96,
        network_dim=128,
        minimum_log_std=-5.0,
        maximum_log_std=1.5,
    ).to(device=device, dtype=dtype)

    return target, flow


def timed(function, /, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def scalar(result: dict[str, torch.Tensor], name: str) -> float:
    return float(result[name].detach().cpu())


def compare_at_threshold(
    *,
    threshold: float,
    flow: AutoregressiveEventFlow,
    target: RareEventTargetDensity,
    importance_samples: int,
    naive_samples: int,
    original_fraction: float,
    confidence_level: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Estimate the same probability by both methods at one threshold."""
    importance, importance_seconds = timed(
        estimate_probability_defensive_mixture,
        flow,
        target,
        number_of_samples=importance_samples,
        original_fraction=original_fraction,
        device=device,
        dtype=dtype,
        confidence_level=confidence_level,
    )
    naive, naive_seconds = timed(
        estimate_probability_naive_monte_carlo,
        target,
        number_of_samples=naive_samples,
        device=device,
        dtype=dtype,
        confidence_level=confidence_level,
    )

    importance_variance = scalar(importance, "estimator_variance")
    naive_variance = scalar(naive, "estimator_variance")
    variance_ratio = (
        naive_variance / importance_variance
        if importance_variance > 0.0
        else float("nan")
    )

    result = {
        "threshold_g": threshold,
        "flow_probability": scalar(importance, "probability"),
        "flow_standard_error": scalar(importance, "standard_error"),
        "flow_relative_error": scalar(importance, "relative_standard_error"),
        "flow_ci_lower": scalar(importance, "confidence_interval_lower"),
        "flow_ci_upper": scalar(importance, "confidence_interval_upper"),
        "flow_exceedance_count": scalar(importance, "exceedance_count"),
        "flow_effective_sample_size": scalar(
            importance, "effective_sample_size"
        ),
        "flow_seconds": importance_seconds,
        "naive_probability": scalar(naive, "probability"),
        "naive_standard_error": scalar(naive, "standard_error"),
        "naive_relative_error": scalar(naive, "relative_standard_error"),
        "naive_ci_lower": scalar(naive, "confidence_interval_lower"),
        "naive_ci_upper": scalar(naive, "confidence_interval_upper"),
        "naive_exceedance_count": scalar(naive, "exceedance_count"),
        "naive_seconds": naive_seconds,
        "variance_ratio_naive_over_flow": variance_ratio,
    }

    print(f"\nThreshold = {threshold:.3f} g")
    print(
        "  Defensive flow IS: "
        f"P={result['flow_probability']:.6e}, "
        f"SE={result['flow_standard_error']:.3e}, "
        f"rel.SE={result['flow_relative_error']:.3f}, "
        f"ESS={result['flow_effective_sample_size']:.1f}, "
        f"time={importance_seconds:.2f}s"
    )
    print(
        "  Naive Monte Carlo: "
        f"P={result['naive_probability']:.6e}, "
        f"SE={result['naive_standard_error']:.3e}, "
        f"rel.SE={result['naive_relative_error']:.3f}, "
        f"exceedances={int(result['naive_exceedance_count'])}, "
        f"time={naive_seconds:.2f}s"
    )
    print(
        "  Variance ratio (naive / flow IS): "
        f"{result['variance_ratio_naive_over_flow']:.3g}"
    )
    return result


def write_comparison_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue one rare-event flow across increasing SA(0.2 s) "
            "thresholds and compare against naive Monte Carlo."
        )
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.5, 0.7, 0.9, 1.1),
        help="Strictly increasing SA(0.2 s) thresholds in g.",
    )
    parser.add_argument("--exposure-time", type=float, default=1.0)
    parser.add_argument("--pretrain-updates", type=int, default=20)
    parser.add_argument("--pretrain-samples", type=int, default=64)
    parser.add_argument("--updates-per-threshold", type=int, default=10)
    parser.add_argument("--samples-per-update", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--mixture-fraction", type=float, default=0.20)
    parser.add_argument("--penalty-alpha", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=1.25)
    parser.add_argument("--importance-samples", type=int, default=2000)
    parser.add_argument("--naive-samples", type=int, default=5000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--plot-candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("progressive_sa_0p2_checkpoint.pt"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume the flow and Adam state from a previous checkpoint.",
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("progressive_sa_0p2_comparison.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("progressive_sa_0p2_sequences.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = tuple(float(value) for value in args.thresholds)

    if any(value <= 0.0 for value in thresholds):
        raise ValueError("All thresholds must be positive.")
    if any(
        current <= previous
        for previous, current in zip(thresholds, thresholds[1:], strict=False)
    ):
        raise ValueError("--thresholds must be strictly increasing.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    dtype = torch.float64

    target, flow = build_problem(
        device=device,
        dtype=dtype,
        exposure_time=args.exposure_time,
        initial_threshold=thresholds[0],
        penalty_alpha=args.penalty_alpha,
    )
    trainer = ProgressiveThresholdTrainer(
        flow,
        target,
        threshold_name="SA_0p2",
        learning_rate=args.learning_rate,
    )

    prior_results: list[dict[str, float]] = []
    if args.resume is not None:
        extra = trainer.load_checkpoint(args.resume, map_location=device)
        prior_results = list(extra.get("threshold_results", []))
        current = trainer.current_threshold
        thresholds_to_run = tuple(value for value in thresholds if value > current)
        print(f"Resumed checkpoint at threshold {current:.3f} g.")
        if not thresholds_to_run:
            print("No requested threshold is higher than the checkpoint threshold.")
    else:
        thresholds_to_run = thresholds
        trainer.pretrain_on_original_model(
            number_of_updates=args.pretrain_updates,
            samples_per_update=args.pretrain_samples,
            device=device,
            dtype=dtype,
        )

    results = prior_results
    for threshold in thresholds_to_run:
        print(f"\nTraining continuation for threshold {threshold:.3f} g")
        trainer.train_threshold(
            threshold,
            number_of_updates=args.updates_per_threshold,
            samples_per_update=args.samples_per_update,
            original_fraction=args.mixture_fraction,
            penalty_alpha=args.penalty_alpha,
            temperature=args.temperature,
            device=device,
            dtype=dtype,
        )

        result = compare_at_threshold(
            threshold=threshold,
            flow=flow,
            target=target,
            importance_samples=args.importance_samples,
            naive_samples=args.naive_samples,
            original_fraction=args.mixture_fraction,
            confidence_level=args.confidence_level,
            device=device,
            dtype=dtype,
        )
        results.append(result)

        trainer.save_checkpoint(
            args.checkpoint,
            extra={
                "threshold_results": results,
                "exposure_time": args.exposure_time,
                "variable_names": flow.variable_names,
            },
        )
        print(f"  Saved resumable checkpoint: {args.checkpoint}")

    if not results:
        raise RuntimeError("No threshold results are available.")

    write_comparison_csv(results, args.comparison_csv)
    print(f"\nSaved threshold comparison: {args.comparison_csv}")

    final_threshold = trainer.current_threshold
    naive_sample, naive_exceeds = select_representative_sequence(
        target,
        method="naive",
        number_of_candidates=args.plot_candidates,
        prefer_exceedance=True,
        device=device,
        dtype=dtype,
    )
    flow_sample, flow_exceeds = select_representative_sequence(
        target,
        method="flow",
        flow=flow,
        number_of_candidates=args.plot_candidates,
        prefer_exceedance=True,
        device=device,
        dtype=dtype,
    )

    figure_path = plot_sequence_comparison(
        naive_sample,
        flow_sample,
        ground_motion_name="SA_0p2",
        threshold=final_threshold,
        output_path=args.figure,
        exposure_time=args.exposure_time,
        title=(
            f"Representative sequences at SA(0.2 s) threshold "
            f"{final_threshold:g} g"
        ),
    )
    print(f"Saved sequence comparison figure: {figure_path}")
    print(
        "  Naive sequence exceeds threshold: "
        f"{naive_exceeds}; events={naive_sample.source.number_of_events}"
    )
    print(
        "  Flow sequence exceeds threshold: "
        f"{flow_exceeds}; events={flow_sample.source.number_of_events}"
    )


if __name__ == "__main__":
    main()
