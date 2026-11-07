from __future__ import annotations

from collections.abc import Mapping
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from models import ColumnPairModel, load_point_feature_config
from project_config import load_project_config

from .config import SupervisedConfig
from .dataset import SupervisedDenoisingDataset, collate_supervised_batch
from .losses import compute_supervised_loss
from .metrics import RunningLossMetrics
from .rollout_eval import run_rollout_evaluation
from .types import EncodedSolutionRecord

_MODEL_CONFIG_KEYS = {
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "dropout",
    "use_global_token",
    "use_value_head",
    "score_dim",
    "state_head_type",
    "time_embedding_type",
}


def _load_model_config_kwargs(config_path: str | Path | None) -> dict[str, Any]:
    """Load optional ``ColumnPairModel`` kwargs from the root ``model`` section.

    ``config_path`` optionally overrides the root project config file. The
    function returns a shallow keyword-argument mapping, does not mutate
    external state, and assumes ``model`` is absent or a mapping
    containing only supported ``ColumnPairModel`` constructor keys, including
    the optional ``state_head_type`` selector.
    """
    loaded = load_project_config(config_path)
    model_config = loaded.get("model", {})
    if model_config is None:
        return {}
    if not isinstance(model_config, Mapping):
        raise ValueError("model config must be a mapping")

    unknown_keys = sorted(set(model_config) - _MODEL_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown model config keys: {unknown_keys}")
    return dict(model_config)


def build_model_from_project_config(config: SupervisedConfig) -> ColumnPairModel:
    """Build the denoising model from validated supervised and project config.

    ``config`` supplies validated supervised settings and may reference a root
    project YAML file containing top-level ``model`` overrides. The function
    returns a ``ColumnPairModel``, does not mutate caller state, and assumes
    any configured model overrides are compatible with the model constructor.
    """
    model_kwargs = _load_model_config_kwargs(config.config_path)

    return ColumnPairModel(
        n=config.n,
        num_steps=2 * config.n + 1,
        point_feature_config=load_point_feature_config(config.config_path),
        **model_kwargs,
    )


def build_optimizer(model: ColumnPairModel, config: SupervisedConfig) -> Optimizer:
    """Build the optimizer used for supervised denoising training."""

    return AdamW(
        model.parameters(),
        lr=config.trainer.lr,
        weight_decay=config.trainer.weight_decay,
    )


class SupervisedTrainer:
    """Train the column-pair model on full and partial denoising tasks."""

    def __init__(
        self,
        config: SupervisedConfig,
        model: ColumnPairModel | None = None,
        optimizer: Optimizer | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the trainer, model, optimizer, and runtime device."""

        self.config = config
        resolved_device = config.runtime.device if device is None else device
        if str(resolved_device) == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved_device)
        self.model = build_model_from_project_config(config) if model is None else model
        self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, config) if optimizer is None else optimizer
        self.latest_metrics: dict[str, float] = {}
        self.latest_rollout_summary: dict[str, object] | None = None
        self._active_stage_name = ""
        self._active_stage_epoch_offset = 0
        self._active_total_epochs = self.config.curriculum.stage1_epochs + self.config.curriculum.stage2_epochs

    def _format_progress_bar(self, completed: int, total: int, width: int = 24) -> str:
        """Return a fixed-width ASCII progress bar for one training epoch."""

        if total <= 0:
            return "[" + "-" * width + "]"
        filled = min(width, round(width * completed / total))
        return "[" + "=" * filled + "-" * (width - filled) + "]"

    def _print_train_progress(
        self,
        *,
        epoch_index: int,
        batch_index: int,
        num_batches: int,
        processed_samples: int,
        total_samples: int,
        current_loss: float,
    ) -> None:
        """Render one in-place progress update when stdout is an interactive TTY."""

        global_epoch_index = self._active_stage_epoch_offset + epoch_index + 1
        progress_bar = self._format_progress_bar(processed_samples, total_samples)
        if not sys.stdout.isatty():
            return
        message = (
            f"\r\033[K{self._active_stage_name} | epoch {global_epoch_index}/{max(self._active_total_epochs, 1)} "
            f"| batch {batch_index}/{num_batches} {progress_bar} "
            f"{processed_samples}/{total_samples} samples | loss {current_loss:.4f}"
        )
        sys.stdout.write(message)
        sys.stdout.flush()

    def _print_epoch_summary(self, stage_name: str, epoch_index: int, metrics: dict[str, float]) -> None:
        """Print one newline-terminated summary line after an epoch finishes."""

        global_epoch_index = self._active_stage_epoch_offset + epoch_index + 1
        summary_parts = [
            f"{stage_name} | epoch {global_epoch_index}/{max(self._active_total_epochs, 1)} complete",
            f"train_loss {metrics['loss']:.4f}",
        ]
        if "val_loss" in metrics:
            summary_parts.append(f"val_loss {metrics['val_loss']:.4f}")
        print(" | ".join(summary_parts), flush=True)

    def _format_rollout_mode_summary(self, label: str, summary: Mapping[str, object]) -> str:
        """Return one concise terminal summary line for a rollout-eval result."""

        if "skipped_reason" in summary:
            return f"rollout_eval | {label} | skipped {summary['skipped_reason']}"

        return (
            f"rollout_eval | {label} | episodes {int(summary['num_episodes'])} "
            f"| mean_final {float(summary['mean_final_point_count']):.2f} "
            f"| mean_gain {float(summary['mean_gain']):.2f} "
            f"| mean_accept_rate {float(summary['mean_accept_rate']):.3f}"
        )

    def _print_rollout_summary(self, rollout_summary: Mapping[str, object]) -> None:
        """Print concise rollout-eval summaries for the configured eval modes."""

        if "skipped_reason" in rollout_summary:
            print(f"rollout_eval | skipped {rollout_summary['skipped_reason']}", flush=True)
            return

        printed_any = False
        for mode_name in ("standard", "partial_state"):
            mode_summary = rollout_summary.get(mode_name)
            if isinstance(mode_summary, Mapping):
                print(self._format_rollout_mode_summary(mode_name, mode_summary), flush=True)
                printed_any = True

        decode_ablation_summary = rollout_summary.get("decode_ablation")
        if isinstance(decode_ablation_summary, Mapping):
            ablations = decode_ablation_summary.get("ablations")
            if isinstance(ablations, Mapping):
                for ablation_name in sorted(ablations):
                    ablation_summary = ablations[ablation_name]
                    if isinstance(ablation_summary, Mapping):
                        print(
                            self._format_rollout_mode_summary(
                                f"decode_ablation/{ablation_name}",
                                ablation_summary,
                            ),
                            flush=True,
                        )
                        printed_any = True

        if not printed_any:
            print("rollout_eval | no_results", flush=True)

    def _build_loader(
        self,
        records: list[EncodedSolutionRecord],
        split_name: str,
        mode: str,
        partial_prob: float,
        shuffle: bool,
    ) -> DataLoader:
        """Build a DataLoader for one training or validation split."""

        dataset = SupervisedDenoisingDataset(
            records=records,
            config=self.config,
            split_name=split_name,
            seed=self.config.runtime.seed,
            mode=mode,
            partial_prob=partial_prob,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.runtime.batch_size,
            shuffle=shuffle,
            num_workers=self.config.runtime.num_workers,
            collate_fn=collate_supervised_batch,
        )

    def fit(
        self,
        train_records: list[EncodedSolutionRecord],
        val_records: list[EncodedSolutionRecord] | None = None,
    ) -> None:
        """Run the configured two-stage supervised training curriculum."""

        if self.config.curriculum.stage1_epochs > 0:
            train_loader = self._build_loader(train_records, "train", "full", 0.0, shuffle=True)
            val_loader = (
                self._build_loader(val_records, "val", "full", 0.0, shuffle=False)
                if val_records
                else None
            )
            self.fit_loader(
                train_loader,
                stage_name="stage1_full",
                num_epochs=self.config.curriculum.stage1_epochs,
                val_loader=val_loader,
                rollout_eval_records=val_records,
                epoch_offset=0,
                total_epochs=self.config.curriculum.stage1_epochs + self.config.curriculum.stage2_epochs,
                save_checkpoints=True,
            )
        if self.config.curriculum.stage2_epochs > 0:
            train_loader = self._build_loader(
                train_records,
                "train",
                self.config.curriculum.stage2_mode,
                self.config.curriculum.stage2_partial_prob,
                shuffle=True,
            )
            val_loader = (
                self._build_loader(
                    val_records,
                    "val",
                    self.config.curriculum.stage2_mode,
                    self.config.curriculum.stage2_partial_prob,
                    shuffle=False,
                )
                if val_records
                else None
            )
            self.fit_loader(
                train_loader,
                stage_name="stage2",
                num_epochs=self.config.curriculum.stage2_epochs,
                val_loader=val_loader,
                rollout_eval_records=val_records,
                epoch_offset=self.config.curriculum.stage1_epochs,
                total_epochs=self.config.curriculum.stage1_epochs + self.config.curriculum.stage2_epochs,
                save_checkpoints=True,
            )

    def fit_loader(
        self,
        train_loader: DataLoader,
        *,
        stage_name: str,
        num_epochs: int,
        val_loader: DataLoader | None = None,
        rollout_eval_records: list[EncodedSolutionRecord] | None = None,
        epoch_offset: int = 0,
        total_epochs: int | None = None,
        save_checkpoints: bool = True,
    ) -> dict[str, float]:
        """Train on a caller-provided DataLoader using supervised loss logic.

        ``train_loader`` yields batches compatible with
        ``collate_supervised_batch`` and ``val_loader`` may provide matching
        validation batches. ``stage_name`` names progress/checkpoints,
        ``num_epochs`` controls training length, ``rollout_eval_records`` are
        optional clean states used by rollout evaluation, and ``epoch_offset``
        preserves curriculum-style global epoch numbering. ``total_epochs``
        overrides the progress denominator. The method returns latest metrics,
        mutates model and optimizer state, and assumes any dataset with a
        ``set_epoch`` method accepts zero-based epoch indices.
        """

        self._active_stage_name = stage_name
        self._active_stage_epoch_offset = epoch_offset
        self._active_total_epochs = epoch_offset + num_epochs if total_epochs is None else total_epochs

        for epoch_index in range(num_epochs):
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch_index)
            train_metrics = self.train_epoch(train_loader, epoch_index)
            self.latest_metrics = dict(train_metrics)

            if val_loader is not None and (epoch_index + 1) % self.config.trainer.eval_every == 0:
                if hasattr(val_loader.dataset, "set_epoch"):
                    val_loader.dataset.set_epoch(epoch_index)
                eval_metrics = self.evaluate_epoch(val_loader, epoch_index)
                self.latest_metrics.update({f"val_{key}": value for key, value in eval_metrics.items()})

            if self.config.trainer.rollout_eval_every > 0 and (epoch_index + 1) % self.config.trainer.rollout_eval_every == 0:
                self.latest_rollout_summary = run_rollout_evaluation(
                    self.model,
                    self.config,
                    eval_records=rollout_eval_records,
                    device=self.device,
                )
                self._print_rollout_summary(self.latest_rollout_summary)

            if save_checkpoints and (epoch_index + 1) % self.config.trainer.save_every == 0:
                self.save_checkpoint(stage_name, epoch_index, self.latest_metrics)
            self._print_epoch_summary(stage_name, epoch_index, self.latest_metrics)

        return dict(self.latest_metrics)

    def _run_stage(
        self,
        stage_name: str,
        dataset_mode: str,
        partial_prob: float,
        num_epochs: int,
        train_records: list[EncodedSolutionRecord],
        val_records: list[EncodedSolutionRecord] | None,
        epoch_offset: int,
    ) -> None:
        """Run one supervised-training stage for the requested number of epochs."""

        train_loader = self._build_loader(train_records, "train", dataset_mode, partial_prob, shuffle=True)
        val_loader = None
        if val_records:
            val_loader = self._build_loader(val_records, "val", dataset_mode, partial_prob, shuffle=False)
        self.fit_loader(
            train_loader,
            stage_name=stage_name,
            num_epochs=num_epochs,
            val_loader=val_loader,
            rollout_eval_records=val_records,
            epoch_offset=epoch_offset,
            total_epochs=self.config.curriculum.stage1_epochs + self.config.curriculum.stage2_epochs,
            save_checkpoints=True,
        )

    def train_epoch(self, loader: DataLoader, epoch_index: int) -> dict[str, float]:
        """Train for one epoch on batched supervised denoising samples.

        ``loader`` yields dictionaries from ``collate_supervised_batch`` whose
        CPU tensor entries have shapes ``[B, S]`` for state ids, token weights,
        and supervision masks, plus ``[B]`` tensors for step ids and current
        point counts. ``epoch_index`` is the zero-based epoch number within the
        active curriculum stage. The method returns epoch-mean loss metrics as
        plain floats, mutates model and optimizer state through gradient
        updates, and assumes each batch's ``task_types`` list matches the
        tensor batch dimension.
        """

        self.model.train()
        metrics = RunningLossMetrics()
        total_samples = len(loader.dataset)
        num_batches = len(loader)
        processed_samples = 0

        for batch_index, batch in enumerate(loader, start=1):
            input_state_ids = batch["input_state_ids"].to(self.device)
            target_state_ids = batch["target_state_ids"].to(self.device)
            current_point_counts = batch["current_point_counts"].to(self.device)
            token_weights = batch["token_weights"].to(self.device)
            supervision_mask = batch["supervision_mask"].to(self.device)
            task_types = batch["task_types"]
            progress_ids = input_state_ids.ne(0).sum(dim=1).to(dtype=torch.long)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(input_state_ids, progress_ids, current_point_counts)
            loss_dict = compute_supervised_loss(
                state_logits=outputs["state_logits"],
                target_state_ids=target_state_ids,
                token_weights=token_weights,
                supervision_mask=supervision_mask,
                task_types=task_types,
                config=self.config.loss,
            )
            loss_dict["loss"].backward()
            if self.config.trainer.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.grad_clip_norm)
            self.optimizer.step()
            batch_size = input_state_ids.shape[0]
            metrics.update(loss_dict, batch_size=batch_size, task_types=task_types)
            processed_samples += batch_size
            self._print_train_progress(
                epoch_index=epoch_index,
                batch_index=batch_index,
                num_batches=num_batches,
                processed_samples=processed_samples,
                total_samples=total_samples,
                current_loss=float(loss_dict["loss"]),
            )

        if num_batches > 0 and sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()

        return metrics.compute()

    def evaluate_epoch(self, loader: DataLoader, epoch_index: int) -> dict[str, float]:
        """Evaluate one epoch on batched supervised denoising samples.

        ``loader`` yields the same batch structure as training, and
        ``epoch_index`` is accepted for API symmetry but otherwise unused. The
        method returns epoch-mean loss metrics as plain floats, does not update
        model parameters or optimizer state apart from switching the model into
        evaluation mode, and assumes each batch's ``task_types`` list matches
        the tensor batch dimension.
        """

        del epoch_index
        self.model.eval()
        metrics = RunningLossMetrics()

        with torch.no_grad():
            for batch in loader:
                input_state_ids = batch["input_state_ids"].to(self.device)
                target_state_ids = batch["target_state_ids"].to(self.device)
                current_point_counts = batch["current_point_counts"].to(self.device)
                token_weights = batch["token_weights"].to(self.device)
                supervision_mask = batch["supervision_mask"].to(self.device)
                task_types = batch["task_types"]
                progress_ids = input_state_ids.ne(0).sum(dim=1).to(dtype=torch.long)

                outputs = self.model(input_state_ids, progress_ids, current_point_counts)
                loss_dict = compute_supervised_loss(
                    state_logits=outputs["state_logits"],
                    target_state_ids=target_state_ids,
                    token_weights=token_weights,
                    supervision_mask=supervision_mask,
                    task_types=task_types,
                    config=self.config.loss,
                )
                metrics.update(loss_dict, batch_size=input_state_ids.shape[0], task_types=task_types)

        return metrics.compute()

    def save_checkpoint(
        self,
        stage_name: str,
        epoch_index: int,
        extra_metrics: dict[str, float],
    ) -> Path:
        """Save a training checkpoint for the current stage and epoch.

        ``stage_name`` and ``epoch_index`` determine the output filename, while
        ``extra_metrics`` is copied into the checkpoint payload. The method
        writes one file, returns its path, and mutates only the filesystem.
        """

        checkpoint_dir = Path(self.config.runtime.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{stage_name}_epoch_{epoch_index + 1}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch_index + 1,
                "stage_name": stage_name,
                "config": asdict(self.config),
                "metrics": dict(extra_metrics),
                "rollout_summary": self.latest_rollout_summary,
            },
            checkpoint_path,
        )
        return checkpoint_path
