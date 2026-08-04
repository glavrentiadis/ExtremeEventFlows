from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .flow import AutoregressiveEventFlow
from .performance import JointGroundMotionThresholdPerformance
from .target import RareEventTargetDensity


class ProgressiveThresholdTrainer:
    """Continue training one flow across successively higher thresholds.

    The flow parameters and Adam optimizer state are retained when the
    threshold changes. This lets a proposal trained for an easier event serve
    as the initialization for a rarer event without repeating original-model
    pretraining.

    Parameters
    ----------
    flow
        Autoregressive proposal to update.
    target
        Rare-event target whose performance threshold is changed in place.
    threshold_name
        Ground-motion output whose threshold is continued, for example
        ``"SA_0p2"``.
    learning_rate
        Adam learning rate.
    weight_decay
        Adam weight decay.
    gradient_clip_norm
        Maximum gradient norm applied after every update.
    """

    def __init__(
        self,
        flow: AutoregressiveEventFlow,
        target: RareEventTargetDensity,
        *,
        threshold_name: str,
        learning_rate: float = 2.0e-4,
        weight_decay: float = 1.0e-6,
        gradient_clip_norm: float = 5.0,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")

        if not isinstance(
            target.performance_function,
            JointGroundMotionThresholdPerformance,
        ):
            raise TypeError(
                "ProgressiveThresholdTrainer requires "
                "JointGroundMotionThresholdPerformance."
            )
        if threshold_name not in target.performance_function.names:
            raise KeyError(
                f"Unknown threshold name {threshold_name!r}. Available names: "
                f"{target.performance_function.names}."
            )

        self.flow = flow
        self.target = target
        self.threshold_name = threshold_name
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.optimizer = torch.optim.Adam(
            flow.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.history: list[dict[str, float]] = []
        self.pretraining_updates_completed = 0

    @property
    def performance_function(self) -> JointGroundMotionThresholdPerformance:
        performance = self.target.performance_function
        assert isinstance(performance, JointGroundMotionThresholdPerformance)
        return performance

    @property
    def current_threshold(self) -> float:
        return self.performance_function.get_threshold(self.threshold_name)

    def set_threshold(self, value: float) -> None:
        """Update the active physical threshold without rebuilding the target."""
        self.performance_function.set_threshold(self.threshold_name, value)

    def pretrain_on_original_model(
        self,
        *,
        number_of_updates: int,
        samples_per_update: int,
        device: torch.device | str,
        dtype: torch.dtype,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Warm-start ``q_theta`` by maximum likelihood on samples from ``p``."""
        if number_of_updates < 0:
            raise ValueError("number_of_updates cannot be negative.")
        if samples_per_update <= 0:
            raise ValueError("samples_per_update must be positive.")

        records: list[dict[str, float]] = []
        for local_update in range(1, number_of_updates + 1):
            with torch.no_grad():
                samples = [
                    self.target.application.sample_original(
                        self.target.exposure_time,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(samples_per_update)
                ]

            log_q = torch.stack(
                [
                    self.flow.log_prob_one(
                        sample,
                        exposure_time=self.target.exposure_time,
                    )
                    for sample in samples
                ]
            )
            loss = -log_q.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.flow.parameters(),
                max_norm=self.gradient_clip_norm,
            )
            self.optimizer.step()

            self.pretraining_updates_completed += 1
            record = {
                "phase": 0.0,
                "update": float(self.pretraining_updates_completed),
                "threshold": self.current_threshold,
                "loss": float(loss.detach().cpu()),
            }
            self.history.append(record)
            records.append(record)

            if verbose and (
                local_update == 1
                or local_update % 5 == 0
                or local_update == number_of_updates
            ):
                print(
                    f"pretrain={local_update:4d}/{number_of_updates}  "
                    f"negative_log_likelihood={record['loss']: .5e}"
                )

        return records

    def defensive_mixture_update(
        self,
        *,
        number_of_samples: int,
        original_fraction: float,
        temperature: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        r"""Perform one weighted cross-entropy update.

        Candidate sequences are sampled from

        .. math::

            r(x)=(1-\epsilon)q_\theta(x)+\epsilon p(x),

        and weighted by the tempered rare-event target divided by ``r``.
        """
        if number_of_samples <= 0:
            raise ValueError("number_of_samples must be positive.")
        if not 0.0 < original_fraction < 1.0:
            raise ValueError("original_fraction must lie between zero and one.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        log_flow_fraction = torch.log(
            torch.tensor(
                1.0 - original_fraction,
                device=device,
                dtype=dtype,
            )
        )
        log_original_fraction = torch.log(
            torch.tensor(
                original_fraction,
                device=device,
                dtype=dtype,
            )
        )

        samples = []
        log_scores: list[Tensor] = []

        with torch.no_grad():
            for _ in range(number_of_samples):
                draw_from_original = bool(
                    (torch.rand((), device=device) < original_fraction).item()
                )
                if draw_from_original:
                    sample = self.target.application.sample_original(
                        self.target.exposure_time,
                        device=device,
                        dtype=dtype,
                    )
                else:
                    sample = self.flow.sample_one(
                        exposure_time=self.target.exposure_time
                    ).sample

                evaluation = self.target(sample)
                log_p = evaluation.log_original_density
                log_q = self.flow.log_prob_one(
                    sample,
                    exposure_time=self.target.exposure_time,
                )
                log_r = torch.logaddexp(
                    log_flow_fraction + log_q,
                    log_original_fraction + log_p,
                )
                samples.append(sample)
                log_scores.append(
                    (evaluation.log_target_unnormalized - log_r) / temperature
                )

            normalized_weights = torch.softmax(torch.stack(log_scores), dim=0)

        log_q_train = torch.stack(
            [
                self.flow.log_prob_one(
                    sample,
                    exposure_time=self.target.exposure_time,
                )
                for sample in samples
            ]
        )
        loss = -(normalized_weights * log_q_train).sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.flow.parameters(),
            max_norm=self.gradient_clip_norm,
        )
        self.optimizer.step()

        weight_ess = 1.0 / (normalized_weights.square().sum() + 1.0e-30)
        return {
            "loss": loss.detach(),
            "maximum_normalized_weight": normalized_weights.max().detach(),
            "weight_entropy": -(
                normalized_weights * torch.log(normalized_weights + 1.0e-30)
            ).sum().detach(),
            "weight_effective_sample_size": weight_ess.detach(),
        }

    def train_threshold(
        self,
        threshold: float,
        *,
        number_of_updates: int,
        samples_per_update: int,
        original_fraction: float,
        penalty_alpha: float,
        temperature: float,
        device: torch.device | str,
        dtype: torch.dtype,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Continue training at one threshold while retaining optimizer state."""
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")
        if number_of_updates < 0:
            raise ValueError("number_of_updates cannot be negative.")
        if penalty_alpha <= 0.0:
            raise ValueError("penalty_alpha must be positive.")

        self.set_threshold(threshold)
        self.target.penalty_function.alpha = float(penalty_alpha)

        records: list[dict[str, float]] = []
        for local_update in range(1, number_of_updates + 1):
            metrics = self.defensive_mixture_update(
                number_of_samples=samples_per_update,
                original_fraction=original_fraction,
                temperature=temperature,
                device=device,
                dtype=dtype,
            )
            record = {
                "phase": 1.0,
                "update": float(len(self.history) + 1),
                "threshold": float(threshold),
                "penalty_alpha": float(penalty_alpha),
                "temperature": float(temperature),
                "loss": float(metrics["loss"].cpu()),
                "maximum_normalized_weight": float(
                    metrics["maximum_normalized_weight"].cpu()
                ),
                "weight_entropy": float(metrics["weight_entropy"].cpu()),
                "weight_effective_sample_size": float(
                    metrics["weight_effective_sample_size"].cpu()
                ),
            }
            self.history.append(record)
            records.append(record)

            if verbose:
                print(
                    f"threshold={threshold:.3f}g  "
                    f"update={local_update:3d}/{number_of_updates}  "
                    f"loss={record['loss']: .5e}  "
                    f"max_weight={record['maximum_normalized_weight']:.3f}  "
                    f"weight_ess={record['weight_effective_sample_size']:.2f}  "
                    f"entropy={record['weight_entropy']:.3f}"
                )

        return records

    def train_schedule(
        self,
        thresholds: Iterable[float],
        *,
        updates_per_threshold: int,
        samples_per_update: int,
        original_fraction: float,
        penalty_alpha: float,
        temperature: float,
        device: torch.device | str,
        dtype: torch.dtype,
        require_increasing: bool = True,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Train successively across thresholds using one flow and optimizer."""
        threshold_values = tuple(float(value) for value in thresholds)
        if not threshold_values:
            raise ValueError("thresholds cannot be empty.")
        if any(value <= 0.0 for value in threshold_values):
            raise ValueError("Every threshold must be positive.")
        if require_increasing and any(
            current <= previous
            for previous, current in zip(
                threshold_values,
                threshold_values[1:],
                strict=False,
            )
        ):
            raise ValueError("thresholds must be strictly increasing.")

        records: list[dict[str, float]] = []
        for threshold in threshold_values:
            records.extend(
                self.train_threshold(
                    threshold,
                    number_of_updates=updates_per_threshold,
                    samples_per_update=samples_per_update,
                    original_fraction=original_fraction,
                    penalty_alpha=penalty_alpha,
                    temperature=temperature,
                    device=device,
                    dtype=dtype,
                    verbose=verbose,
                )
            )
        return records

    def state_dict(self) -> dict[str, Any]:
        """Return flow, optimizer, threshold, and training-history state."""
        return {
            "flow_state_dict": self.flow.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "threshold_name": self.threshold_name,
            "current_threshold": self.current_threshold,
            "history": list(self.history),
            "pretraining_updates_completed": self.pretraining_updates_completed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore continuation state into an already constructed trainer."""
        checkpoint_name = state.get("threshold_name", self.threshold_name)
        if checkpoint_name != self.threshold_name:
            raise ValueError(
                f"Checkpoint threshold name {checkpoint_name!r} does not match "
                f"{self.threshold_name!r}."
            )
        self.flow.load_state_dict(state["flow_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.set_threshold(float(state["current_threshold"]))
        self.history = list(state.get("history", []))
        self.pretraining_updates_completed = int(
            state.get("pretraining_updates_completed", 0)
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a resumable continuation checkpoint."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_dict()
        if extra is not None:
            payload["extra"] = extra
        torch.save(payload, destination)
        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: torch.device | str | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint and return its optional ``extra`` dictionary."""
        payload = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=False,
        )
        self.load_state_dict(payload)
        return dict(payload.get("extra", {}))
