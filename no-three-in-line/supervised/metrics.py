from __future__ import annotations


class RunningLossMetrics:
    """Accumulate mean batch losses into epoch-level task-aware summaries."""

    def __init__(self) -> None:
        """Initialize empty metric accumulators."""

        self.reset()

    def update(self, loss_dict: dict[str, object], batch_size: int, task_types: list[str]) -> None:
        """Update metric accumulators from one batch of supervised losses."""

        full_count = sum(task_type == "full" for task_type in task_types)
        partial_count = sum(task_type == "partial" for task_type in task_types)

        self.total_loss_sum += float(loss_dict["loss"]) * batch_size
        self.total_count += batch_size
        if full_count > 0:
            self.full_loss_sum += float(loss_dict["mean_full_loss"]) * full_count
            self.full_count += full_count
        if partial_count > 0:
            self.partial_loss_sum += float(loss_dict["mean_partial_loss"]) * partial_count
            self.partial_count += partial_count

    def compute(self) -> dict[str, float]:
        """Return epoch-level mean losses accumulated so far."""

        return {
            "loss": self.total_loss_sum / max(self.total_count, 1),
            "mean_full_loss": self.full_loss_sum / max(self.full_count, 1),
            "mean_partial_loss": self.partial_loss_sum / max(self.partial_count, 1),
            "num_full_samples": float(self.full_count),
            "num_partial_samples": float(self.partial_count),
        }

    def reset(self) -> None:
        """Reset all running totals to zero."""

        self.total_loss_sum = 0.0
        self.total_count = 0
        self.full_loss_sum = 0.0
        self.full_count = 0
        self.partial_loss_sum = 0.0
        self.partial_count = 0
