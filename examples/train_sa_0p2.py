#!/usr/bin/env python3
"""Train an autoregressive importance-sampling flow for an SA(0.2 s) threshold.

The example uses:

* a variable-length Markov source sequence;
* a joint source-event distribution for ``dt``, magnitude, and location;
* a single-output lognormal ground-motion model for ``SA_0p2``;
* weighted cross-entropy updates for the autoregressive proposal;
* defensive-mixture importance sampling and naïve Monte Carlo; and
* a side-by-side uncertainty and runtime comparison.

Run from the package root with, for example::

    python examples/train_sa_0p2.py --threshold-g 0.8

For a very fast smoke run::

    python examples/train_sa_0p2.py \
        --updates-per-stage 1 \
        --samples-per-update 8 \
        --importance-samples 20 \
        --naive-samples 20
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import time

import torch
from torch import Tensor

# Allow the example to run directly from the repository without installing the
# package. This only changes sys.path for the current Python process.
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
    RareEventTargetDensity,
    Site,
    estimate_probability_defensive_mixture,
    estimate_probability_flow_only,
    estimate_probability_naive_monte_carlo,
)


def build_source_model(*, dtype: torch.dtype) -> MarkovJointSourceModel:
    """Construct a joint Markov model for time, magnitude, and source location."""
    event_transforms = {
        "dt": PositiveSoftplusTransform(minimum=0.0),
        "magnitude": BoundedSigmoidTransform(low=4.0, high=8.5),
        "x": BoundedSigmoidTransform(low=-20.0, high=20.0),
        "y": BoundedSigmoidTransform(low=-20.0, high=20.0),
        "depth": BoundedSigmoidTransform(low=5.0, high=15.0),
    }

    # The reference event is also the state used before the first earthquake.
    reference_event = {
        "dt": 0.60,
        "magnitude": 5.0,
        "x": 0.0,
        "y": 0.0,
        "depth": 10.0,
    }

    # Set the latent mean so that the transformed mean is near the reference
    # event. This gives the untrained proposal and physical source comparable
    # scales at initialization.
    base_mean = torch.stack(
        [
            event_transforms[name].inverse(
                torch.tensor(reference_event[name], dtype=dtype)
            )
            for name in event_transforms
        ]
    )

    # A modest first-order Markov dependence. Rows correspond to the current
    # latent variables and columns to the standardized previous physical event.
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

    # The lower-triangular matrix defines the joint conditional covariance.
    # The negative (magnitude, dt) entry creates dependence between the current
    # waiting time and current magnitude.
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

    state_scale = {
        "dt": 0.60,
        "magnitude": 1.0,
        "x": 10.0,
        "y": 10.0,
        "depth": 5.0,
    }

    kernel = GaussianLatentMarkovKernel(
        event_transforms=event_transforms,
        time_increment_name="dt",
        base_mean=base_mean,
        transition_matrix=transition_matrix,
        latent_cholesky=latent_cholesky,
        reference_event=reference_event,
        state_scale=state_scale,
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
    threshold_g: float,
    initial_penalty_alpha: float,
) -> tuple[RareEventTargetDensity, AutoregressiveEventFlow]:
    """Build the physical target and trainable proposal flow."""
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

    # ln SA(0.2 s) = b0 + bM M + bR ln(R + 5) + bV ln(Vs30 / 500)
    # The coefficients are illustrative. Replace them with a calibrated GMM or
    # differentiable surrogate for scientific use.
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

    # For one output, mode="all" and mode="any" are equivalent. The
    # performance is max_i log(SA_0p2_i / 0.5), so exceedance corresponds to
    # performance >= 0.
    performance = JointGroundMotionThresholdPerformance(
        thresholds={"SA_0p2": threshold_g},
        mode="all",
    )

    target = RareEventTargetDensity(
        application=application,
        performance_function=performance,
        penalty_function=PenaltyFunction(
            threshold=0.0,
            alpha=initial_penalty_alpha,
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



def update_sa_threshold(
    target: RareEventTargetDensity,
    threshold_g: float,
) -> None:
    """Update the SA(0.2 s) threshold without rebuilding the problem."""
    if threshold_g <= 0.0:
        raise ValueError("threshold_g must be positive.")

    performance = target.performance_function
    if not isinstance(performance, JointGroundMotionThresholdPerformance):
        raise TypeError(
            "The target does not use JointGroundMotionThresholdPerformance."
        )

    performance.set_threshold("SA_0p2", threshold_g)


def pretrain_on_original_model(
    flow: AutoregressiveEventFlow,
    target: RareEventTargetDensity,
    *,
    number_of_updates: int,
    samples_per_update: int,
    learning_rate: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Warm-start q_theta by maximum likelihood on samples from p.

    This step is important because cross-entropy rare-event adaptation assumes
    that the proposal already has reasonable overlap with the original model.
    """
    if number_of_updates <= 0:
        return

    optimizer = torch.optim.Adam(
        flow.parameters(),
        lr=learning_rate,
        weight_decay=1.0e-6,
    )

    for update in range(1, number_of_updates + 1):
        with torch.no_grad():
            samples = [
                target.application.sample_original(
                    target.exposure_time,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(samples_per_update)
            ]

        log_q = torch.stack(
            [
                flow.log_prob_one(
                    sample,
                    exposure_time=target.exposure_time,
                )
                for sample in samples
            ]
        )
        loss = -log_q.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
        optimizer.step()

        if update == 1 or update % 5 == 0 or update == number_of_updates:
            print(
                f"pretrain={update:4d}/{number_of_updates}  "
                f"negative_log_likelihood={float(loss.detach().cpu()): .5e}"
            )


def mixture_cross_entropy_update(
    flow: AutoregressiveEventFlow,
    target: RareEventTargetDensity,
    optimizer: torch.optim.Optimizer,
    *,
    number_of_samples: int,
    original_fraction: float,
    temperature: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Robust cross-entropy update using a flow/original mixture.

    The candidate density is

        r(x) = (1 - epsilon) q_theta(x) + epsilon p(x).

    Sampling some candidates from the original model prevents the adaptive
    proposal from losing support before it has learned the physical density.
    """
    if not 0.0 < original_fraction < 1.0:
        raise ValueError("original_fraction must lie between zero and one.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    samples = []
    log_scores = []
    log_one_minus = torch.log(
        torch.tensor(1.0 - original_fraction, device=device, dtype=dtype)
    )
    log_original_fraction = torch.log(
        torch.tensor(original_fraction, device=device, dtype=dtype)
    )

    with torch.no_grad():
        for _ in range(number_of_samples):
            if bool((torch.rand((), device=device) < original_fraction).item()):
                sample = target.application.sample_original(
                    target.exposure_time,
                    device=device,
                    dtype=dtype,
                )
            else:
                sample = flow.sample_one(
                    exposure_time=target.exposure_time
                ).sample

            evaluation = target(sample)
            log_p = evaluation.log_original_density
            log_q = flow.log_prob_one(
                sample,
                exposure_time=target.exposure_time,
            )
            log_r = torch.logaddexp(
                log_one_minus + log_q,
                log_original_fraction + log_p,
            )
            samples.append(sample)
            log_scores.append(
                (evaluation.log_target_unnormalized - log_r) / temperature
            )

        normalized_weights = torch.softmax(torch.stack(log_scores), dim=0)

    log_q_train = torch.stack(
        [
            flow.log_prob_one(
                sample,
                exposure_time=target.exposure_time,
            )
            for sample in samples
        ]
    )
    loss = -(normalized_weights * log_q_train).sum()

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



def train_flow(
    flow: AutoregressiveEventFlow,
    target: RareEventTargetDensity,
    *,
    updates_per_stage: int,
    samples_per_update: int,
    learning_rate: float,
    original_fraction: float,
    final_threshold_g: float,
    threshold_schedule: tuple[float, ...] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, float]]:
    """Train using staged weighted cross-entropy updates."""
    optimizer = torch.optim.Adam(
        flow.parameters(),
        lr=learning_rate,
        weight_decay=1.0e-6,
    )

    # Increasing alpha progressively sharpens the smooth rare-event target.
    stages = (
        (5.0, 2.0),
        (10.0, 1.5),
        (20.0, 1.0),
        (35.0, 0.75),
    )

    if threshold_schedule is None:
        stage_thresholds = (final_threshold_g,) * len(stages)
    else:
        stage_thresholds = tuple(float(value) for value in threshold_schedule)
        if len(stage_thresholds) != len(stages):
            raise ValueError(
                f"threshold_schedule must contain exactly {len(stages)} values."
            )
        if any(value <= 0.0 for value in stage_thresholds):
            raise ValueError("All scheduled thresholds must be positive.")

    history: list[dict[str, float]] = []
    global_update = 0

    for (alpha, temperature), threshold_g in zip(
        stages,
        stage_thresholds,
        strict=True,
    ):
        target.penalty_function.alpha = alpha
        update_sa_threshold(target, threshold_g)

        for stage_update in range(1, updates_per_stage + 1):
            global_update += 1
            metrics = mixture_cross_entropy_update(
                flow,
                target,
                optimizer,
                number_of_samples=samples_per_update,
                original_fraction=original_fraction,
                temperature=temperature,
                device=device,
                dtype=dtype,
            )

            record = {
                "update": float(global_update),
                "alpha": alpha,
                "temperature": temperature,
                "threshold_g": threshold_g,
                "loss": float(metrics["loss"].cpu()),
                "maximum_normalized_weight": float(
                    metrics["maximum_normalized_weight"].cpu()
                ),
                "weight_entropy": float(metrics["weight_entropy"].cpu()),
            }
            history.append(record)

            print(
                f"update={global_update:4d}  "
                f"stage={stage_update:3d}/{updates_per_stage}  "
                f"threshold={threshold_g:.3f}g  "
                f"alpha={alpha:5.1f}  "
                f"temperature={temperature:4.2f}  "
                f"loss={record['loss']: .5e}  "
                f"max_weight={record['maximum_normalized_weight']:.3f}  "
                f"entropy={record['weight_entropy']:.3f}"
            )

    return history


def print_metrics(title: str, metrics: dict[str, Tensor]) -> None:
    print(f"\n{title}")
    for name, value in metrics.items():
        print(f"  {name}: {float(value.detach().cpu()):.6g}")


def run_timed_estimator(function, /, *args, **kwargs) -> tuple[dict[str, Tensor], float]:
    """Run one estimator and return its wall-clock time in seconds."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def print_estimator_comparison(
    importance_result: dict[str, Tensor],
    importance_seconds: float,
    naive_result: dict[str, Tensor],
    naive_seconds: float,
) -> None:
    """Print a side-by-side statistical comparison of both estimators."""

    def scalar(result: dict[str, Tensor], name: str) -> float:
        return float(result[name].detach().cpu())

    rows = (
        ("Defensive flow IS", importance_result, importance_seconds),
        ("Naive Monte Carlo", naive_result, naive_seconds),
    )

    print("\nEstimator comparison")
    print(
        f"{'method':<22}"
        f"{'samples':>10}"
        f"{'exceed.':>10}"
        f"{'estimate':>14}"
        f"{'std. error':>14}"
        f"{'rel. error':>14}"
        f"{'95% CI':>27}"
        f"{'seconds':>11}"
    )
    print("-" * 122)

    for method, result, elapsed in rows:
        probability = scalar(result, "probability")
        standard_error = scalar(result, "standard_error")
        relative_error = scalar(result, "relative_standard_error")
        ci_lower = scalar(result, "confidence_interval_lower")
        ci_upper = scalar(result, "confidence_interval_upper")
        sample_count = int(scalar(result, "number_of_samples"))
        exceedance_count = int(scalar(result, "exceedance_count"))

        confidence_interval = f"[{ci_lower:.3e}, {ci_upper:.3e}]"
        print(
            f"{method:<22}"
            f"{sample_count:>10d}"
            f"{exceedance_count:>10d}"
            f"{probability:>14.6e}"
            f"{standard_error:>14.6e}"
            f"{relative_error:>14.6e}"
            f"{confidence_interval:>27}"
            f"{elapsed:>11.3f}"
        )

    importance_variance = scalar(importance_result, "estimator_variance")
    naive_variance = scalar(naive_result, "estimator_variance")
    if importance_variance > 0.0 and naive_variance > 0.0:
        variance_reduction = naive_variance / importance_variance
        print(
            "\nEstimated variance reduction "
            f"(naive variance / flow-IS variance): {variance_reduction:.3g}"
        )
    else:
        print(
            "\nEstimated variance reduction is undefined because one "
            "estimator returned zero empirical variance."
        )

    print(
        "The variance ratio is a per-run diagnostic; it does not by itself "
        "account for the additional cost of evaluating the flow density."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a rare-event flow for a configurable SA(0.2 s) threshold."
    )
    parser.add_argument("--exposure-time", type=float, default=1.0)
    parser.add_argument(
        "--threshold-g",
        type=float,
        default=0.5,
        help="Final SA(0.2 s) exceedance threshold in g.",
    )
    parser.add_argument(
        "--threshold-schedule",
        type=float,
        nargs=4,
        metavar=("G1", "G2", "G3", "G4"),
        default=None,
        help=(
            "Optional four-stage threshold curriculum in g. The final value "
            "must equal --threshold-g. Without this option, every stage uses "
            "--threshold-g."
        ),
    )
    parser.add_argument("--pretrain-updates", type=int, default=20)
    parser.add_argument("--pretrain-samples", type=int, default=64)
    parser.add_argument("--updates-per-stage", type=int, default=10)
    parser.add_argument("--samples-per-update", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--mixture-fraction", type=float, default=0.20)
    parser.add_argument("--importance-samples", type=int, default=1000)
    parser.add_argument(
        "--naive-samples",
        "--crude-samples",
        dest="naive_samples",
        type=int,
        default=2000,
        help="Number of direct samples from the physical model.",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trained_sa_0p2_flow.pt"),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.threshold_g <= 0.0:
        raise ValueError("--threshold-g must be positive.")
    if args.threshold_schedule is not None:
        if any(value <= 0.0 for value in args.threshold_schedule):
            raise ValueError("Every --threshold-schedule value must be positive.")
        if not math.isclose(
            args.threshold_schedule[-1],
            args.threshold_g,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "The final --threshold-schedule value must equal --threshold-g."
            )

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
        threshold_g=args.threshold_g,
        initial_penalty_alpha=5.0,
    )

    print("Rare event:")
    print(
        f"  max_i SA_0p2(i) > {args.threshold_g:.3f} g "
        f"during T = {args.exposure_time:g}"
    )
    print(f"  flow event dimension: {flow.event_dimension}")
    print(f"  generated variables: {flow.variable_names}")

    # First fit the proposal to the original physical model. This stabilizes
    # importance weights before adapting the flow toward the rare event.
    pretrain_on_original_model(
        flow,
        target,
        number_of_updates=args.pretrain_updates,
        samples_per_update=args.pretrain_samples,
        learning_rate=args.learning_rate,
        device=device,
        dtype=dtype,
    )

    before = estimate_probability_flow_only(
        flow,
        target,
        number_of_samples=max(20, min(200, args.importance_samples)),
    )
    print_metrics("After original-model pretraining", before)

    history = train_flow(
        flow,
        target,
        updates_per_stage=args.updates_per_stage,
        samples_per_update=args.samples_per_update,
        learning_rate=args.learning_rate,
        original_fraction=args.mixture_fraction,
        final_threshold_g=args.threshold_g,
        threshold_schedule=(
            tuple(args.threshold_schedule)
            if args.threshold_schedule is not None
            else None
        ),
        device=device,
        dtype=dtype,
    )

    # Restore the final event definition and target sharpness before estimation.
    update_sa_threshold(target, args.threshold_g)
    target.penalty_function.alpha = 35.0

    importance_result, importance_seconds = run_timed_estimator(
        estimate_probability_defensive_mixture,
        flow,
        target,
        number_of_samples=args.importance_samples,
        original_fraction=args.mixture_fraction,
        device=device,
        dtype=dtype,
        confidence_level=args.confidence_level,
    )
    print_metrics(
        "Defensive-mixture importance-sampling estimate",
        importance_result,
    )

    flow_only_diagnostic = estimate_probability_flow_only(
        flow,
        target,
        number_of_samples=max(20, min(200, args.importance_samples)),
        confidence_level=args.confidence_level,
    )
    print_metrics("Flow-only diagnostic", flow_only_diagnostic)

    naive_result, naive_seconds = run_timed_estimator(
        estimate_probability_naive_monte_carlo,
        target,
        number_of_samples=args.naive_samples,
        device=device,
        dtype=dtype,
        confidence_level=args.confidence_level,
    )
    print_metrics("Naive Monte Carlo estimate", naive_result)

    print_estimator_comparison(
        importance_result,
        importance_seconds,
        naive_result,
        naive_seconds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "flow_state_dict": flow.state_dict(),
            "threshold_g": args.threshold_g,
            "threshold_schedule": (
                tuple(args.threshold_schedule)
                if args.threshold_schedule is not None
                else None
            ),
            "exposure_time": args.exposure_time,
            "variable_names": flow.variable_names,
            "training_history": history,
            "importance_result": {
                name: value.detach().cpu()
                for name, value in importance_result.items()
            },
            "importance_elapsed_seconds": importance_seconds,
            "naive_result": {
                name: value.detach().cpu()
                for name, value in naive_result.items()
            },
            "naive_elapsed_seconds": naive_seconds,
        },
        args.output,
    )
    print(f"\nSaved trained flow to: {args.output}")


if __name__ == "__main__":
    main()
