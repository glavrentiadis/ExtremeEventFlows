#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extreme_event_flows import (
    AutoregressiveEventFlow,
    BoundedSigmoidTransform,
    GaussianLatentMarkovKernel,
    GroundMotionApplication,
    IdentityTransform,
    JointGroundMotionThresholdPerformance,
    MarkovJointSourceModel,
    MultiOutputGaussianGroundMotionModel,
    PenaltyFunction,
    PointSourceFeatureBuilder,
    PositiveSoftplusTransform,
    RareEventTargetDensity,
    Site,
)


def build_models(*, device: str = "cpu", dtype: torch.dtype = torch.float64):
    # ------------------------------------------------------------------
    # Joint Markov source model: dt, magnitude, and 3-D source location.
    # ------------------------------------------------------------------
    event_transforms = {
        "dt": PositiveSoftplusTransform(minimum=0.0),
        "magnitude": BoundedSigmoidTransform(4.0, 8.5),
        "x": IdentityTransform(),
        "y": IdentityTransform(),
        "depth": BoundedSigmoidTransform(5.0, 15.0),
    }

    base_mean = torch.tensor([-0.50, -1.10, 0.0, 0.0, 0.0])

    # Every row controls the next latent variable; every column is a feature
    # from the preceding physical event. Nonzero terms create Markov memory.
    transition_matrix = torch.tensor(
        [
            [0.00, -0.20, 0.00, 0.00, 0.00],
            [0.10, 0.15, 0.00, 0.00, 0.00],
            [0.00, 0.05, 0.55, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.55, 0.00],
            [0.00, 0.05, 0.00, 0.00, 0.35],
        ]
    )

    # Lower-triangular latent scale. Off-diagonal terms create a joint current
    # distribution among time, magnitude, and source location.
    latent_cholesky = torch.tensor(
        [
            [0.65, 0.00, 0.00, 0.00, 0.00],
            [-0.20, 0.75, 0.00, 0.00, 0.00],
            [0.05, 0.10, 4.00, 0.00, 0.00],
            [0.00, -0.05, 0.50, 3.50, 0.00],
            [0.05, 0.08, 0.00, 0.00, 0.70],
        ]
    )

    source_kernel = GaussianLatentMarkovKernel(
        event_transforms=event_transforms,
        time_increment_name="dt",
        base_mean=base_mean,
        transition_matrix=transition_matrix,
        latent_cholesky=latent_cholesky,
        reference_event={
            "dt": 0.7,
            "magnitude": 5.0,
            "x": 0.0,
            "y": 0.0,
            "depth": 10.0,
        },
        state_scale={
            "dt": 1.0,
            "magnitude": 1.0,
            "x": 10.0,
            "y": 10.0,
            "depth": 5.0,
        },
    )

    source_model = MarkovJointSourceModel(
        name="Joint Markov source",
        joint_kernel=source_kernel,
    )

    # ------------------------------------------------------------------
    # Generic site and three-output ground-motion model.
    # ------------------------------------------------------------------
    site = Site(
        name="Site A",
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
    )

    ground_motion_model = MultiOutputGaussianGroundMotionModel(
        feature_builder=feature_builder,
        output_names=("PGA", "SA_0p2", "SA_1p0"),
        coefficients=torch.tensor(
            [
                [-4.25, 1.15, -1.10, -0.20],
                [-3.90, 1.20, -1.05, -0.25],
                [-4.60, 1.10, -0.95, -0.15],
            ]
        ),
        residual_cholesky=torch.tensor(
            [
                [0.55, 0.00, 0.00],
                [0.30, 0.48, 0.00],
                [0.20, 0.15, 0.50],
            ]
        ),
    )

    application = GroundMotionApplication(
        source_model=source_model,
        site=site,
        ground_motion_model=ground_motion_model,
    ).to(device=device, dtype=dtype)

    # The event must exceed all three thresholds simultaneously. The returned
    # performance is a joint log margin, so the rare-event threshold is zero.
    performance = JointGroundMotionThresholdPerformance(
        {
            "PGA": 0.35,
            "SA_0p2": 0.70,
            "SA_1p0": 0.25,
        },
        mode="all",
    )

    target = RareEventTargetDensity(
        application=application,
        performance_function=performance,
        penalty_function=PenaltyFunction(threshold=0.0, alpha=25.0),
        exposure_time=50.0,
    ).to(device=device, dtype=dtype)

    proposal = AutoregressiveEventFlow(
        application=application,
        hidden_dim=96,
        network_dim=128,
    ).to(device=device, dtype=dtype)

    return application, target, proposal


def main() -> None:
    application, target, proposal = build_models()

    proposal_sample = proposal.sample_one(exposure_time=target.exposure_time)
    evaluation = target(proposal_sample.sample)
    recomputed_log_q = proposal.log_prob_one(
        proposal_sample.sample,
        exposure_time=target.exposure_time,
    )

    print("Number of source variables:", len(application.source_model.event_names))
    print("Source variables:", application.source_model.event_names)
    print("Ground-motion outputs:", application.ground_motion_model.output_names)
    print("Number of events:", proposal_sample.number_of_events)
    print("Performance:", evaluation.performance.item())
    print("Rare event:", bool(evaluation.rare_event_indicator.item()))
    print("log p:", evaluation.log_original_density.item())
    print("log q:", proposal_sample.log_q.item())
    print("log-q consistency error:", (proposal_sample.log_q - recomputed_log_q).item())


if __name__ == "__main__":
    main()
