"""Top-level model wrapper for the circle packing research scaffold."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from circle_pack.config import ProjectConfig
from models.backbone import CircleTransformerBackbone
from models.features import SlotFeatureBuilder
from models.heads import DDPMHead, FMHead, SimpleProposalHead, ValueHead
from models.interfaces import decode_latent_proposal_for_test, validate_model_inputs
from models.model_types import LatentToCircleFn, ModelOutputs


class CirclePackingModel(nn.Module):
    """Shared-slot Transformer wrapper with optional prediction heads.

    All `K` slots are always visible to the backbone. Fixed/free distinctions are
    represented through input role features and optional head-level zeroing.
    """

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.config = config
        self.model_config = config.model
        self.feature_builder = SlotFeatureBuilder(self.model_config.slot_features)
        self.backbone = CircleTransformerBackbone(self.feature_builder.feature_dim, self.model_config)

        self.ddpm_head = (
            DDPMHead(
                self.model_config.d_model,
                target=self.model_config.heads.ddpm_target,
                zero_fixed_outputs=self.model_config.heads.ddpm_zero_fixed_outputs,
            )
            if self.model_config.heads.enable_ddpm_head
            else None
        )
        self.fm_head = (
            FMHead(
                self.model_config.d_model,
                zero_fixed_outputs=self.model_config.heads.fm_zero_fixed_outputs,
            )
            if self.model_config.heads.enable_fm_head
            else None
        )
        self.simple_proposal_head = (
            SimpleProposalHead(
                self.model_config.d_model,
                target=self.model_config.heads.proposal_target,
                zero_fixed_outputs=self.model_config.heads.proposal_zero_fixed_outputs,
            )
            if self.model_config.heads.enable_simple_proposal_head
            else None
        )
        self.value_head = (
            ValueHead(self.model_config.d_model, self.model_config.activation)
            if self.model_config.heads.enable_value_head
            else None
        )

        self.to(dtype=self.config.torch_dtype)

    @property
    def device(self) -> torch.device:
        """Return the current device of the model parameters."""

        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the current floating dtype of the model parameters."""

        return next(self.parameters()).dtype

    def _empty_outputs(self) -> ModelOutputs:
        """Return the canonical output dictionary with all keys present."""

        return {
            "ddpm_pred": None,
            "fm_pred": None,
            "proposal_latent": None,
            "value": None,
            "slot_hidden": None,
            "global_hidden": None,
        }

    def _prepare_inputs(
        self,
        slot_latent: torch.Tensor,
        slot_centers: torch.Tensor,
        fixed_mask: torch.Tensor,
        outer_step: torch.Tensor | None = None,
        inner_time: torch.Tensor | None = None,
        extra_slot_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Validate and cast common model inputs onto the model device and dtype."""

        slot_latent = torch.as_tensor(slot_latent)
        slot_centers = torch.as_tensor(slot_centers)
        fixed_mask = torch.as_tensor(fixed_mask, dtype=torch.bool)
        extra_features = None if extra_slot_features is None else torch.as_tensor(extra_slot_features)
        validate_model_inputs(slot_latent, slot_centers, fixed_mask, extra_features)

        slot_latent = slot_latent.to(device=self.device, dtype=self.dtype)
        slot_centers = slot_centers.to(device=self.device, dtype=self.dtype)
        fixed_mask = fixed_mask.to(device=self.device)
        if outer_step is not None:
            outer_step = torch.as_tensor(outer_step, device=self.device, dtype=self.dtype)
        if inner_time is not None:
            inner_time = torch.as_tensor(inner_time, device=self.device, dtype=self.dtype)
        if extra_features is not None:
            extra_features = extra_features.to(device=self.device, dtype=self.dtype)
        return slot_latent, slot_centers, fixed_mask, outer_step, inner_time, extra_features

    def forward(
        self,
        slot_latent: torch.Tensor,
        slot_centers: torch.Tensor,
        fixed_mask: torch.Tensor,
        outer_step: torch.Tensor | None = None,
        inner_time: torch.Tensor | None = None,
        extra_slot_features: torch.Tensor | None = None,
        *,
        predict_ddpm: bool = False,
        predict_fm: bool = False,
        predict_proposal: bool = False,
        predict_value: bool = False,
        return_hidden: bool = False,
    ) -> ModelOutputs:
        """Run the shared backbone and any requested optional heads.

        Parameters
        ----------
        slot_latent:
            Tensor of shape `[B, K, 2]`.
        slot_centers:
            Tensor of shape `[B, K, 2]`.
        fixed_mask:
            Boolean tensor of shape `[B, K]`.
        outer_step:
            Optional tensor of shape `[B]` or `[B, 1]`.
        inner_time:
            Optional tensor of shape `[B]` or `[B, 1]`.
        extra_slot_features:
            Optional tensor of shape `[B, K, D_extra]`.
        """

        (
            slot_latent,
            slot_centers,
            fixed_mask,
            outer_step,
            inner_time,
            extra_slot_features,
        ) = self._prepare_inputs(
            slot_latent,
            slot_centers,
            fixed_mask,
            outer_step=outer_step,
            inner_time=inner_time,
            extra_slot_features=extra_slot_features,
        )

        raw_slot_features = self.feature_builder(
            slot_latent,
            slot_centers,
            fixed_mask,
            extra_slot_features=extra_slot_features,
        )
        backbone_output = self.backbone(
            raw_slot_features,
            slot_centers,
            fixed_mask,
            outer_step=outer_step,
            inner_time=inner_time,
        )

        outputs = self._empty_outputs()
        slot_hidden = backbone_output.slot_hidden
        global_hidden = backbone_output.global_hidden

        if predict_ddpm:
            if self.ddpm_head is None:
                raise ValueError("DDPM head is disabled in the config.")
            outputs["ddpm_pred"] = self.ddpm_head(slot_hidden, fixed_mask=fixed_mask)
        if predict_fm:
            if self.fm_head is None:
                raise ValueError("FM head is disabled in the config.")
            outputs["fm_pred"] = self.fm_head(slot_hidden, fixed_mask=fixed_mask)
        if predict_proposal:
            if self.simple_proposal_head is None:
                raise ValueError("Simple proposal head is disabled in the config.")
            outputs["proposal_latent"] = self.simple_proposal_head(slot_hidden, fixed_mask=fixed_mask)
        if predict_value:
            if self.value_head is None:
                raise ValueError("Value head is disabled in the config.")
            value_input = global_hidden if global_hidden is not None else slot_hidden.mean(dim=1)
            outputs["value"] = self.value_head(value_input)
        if return_hidden:
            outputs["slot_hidden"] = slot_hidden
            outputs["global_hidden"] = global_hidden
        return outputs

    @torch.no_grad()
    def predict_proposal_latent(
        self,
        slot_latent: torch.Tensor,
        slot_centers: torch.Tensor,
        fixed_mask: torch.Tensor,
        outer_step: torch.Tensor | None = None,
        inner_time: torch.Tensor | None = None,
        extra_slot_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return an absolute latent proposal tensor of shape `[B, K, 2]`."""

        (
            slot_latent,
            slot_centers,
            fixed_mask,
            outer_step,
            inner_time,
            extra_slot_features,
        ) = self._prepare_inputs(
            slot_latent,
            slot_centers,
            fixed_mask,
            outer_step=outer_step,
            inner_time=inner_time,
            extra_slot_features=extra_slot_features,
        )
        outputs = self.forward(
            slot_latent,
            slot_centers,
            fixed_mask,
            outer_step=outer_step,
            inner_time=inner_time,
            extra_slot_features=extra_slot_features,
            predict_proposal=True,
            return_hidden=False,
        )
        raw_proposal = outputs["proposal_latent"]
        if raw_proposal is None:
            raise AssertionError("predict_proposal_latent requires the simple proposal head.")

        if self.model_config.heads.proposal_target == "delta_z":
            proposal_latent = slot_latent + raw_proposal
        else:
            proposal_latent = raw_proposal

        if self.model_config.heads.proposal_target == "z_prop" and self.model_config.heads.proposal_zero_fixed_outputs:
            proposal_latent = torch.where(fixed_mask.unsqueeze(-1), slot_latent, proposal_latent)
        return proposal_latent

    @torch.no_grad()
    def predict_proposal_centers_for_test(
        self,
        slot_latent: torch.Tensor,
        slot_centers: torch.Tensor,
        fixed_mask: torch.Tensor,
        outer_step: torch.Tensor | None = None,
        inner_time: torch.Tensor | None = None,
        extra_slot_features: torch.Tensor | None = None,
        latent_to_center_fn: LatentToCircleFn | None = None,
    ) -> torch.Tensor:
        """Decode proposal latents into proposal centers for tests and demos."""

        proposal_latent = self.predict_proposal_latent(
            slot_latent,
            slot_centers,
            fixed_mask,
            outer_step=outer_step,
            inner_time=inner_time,
            extra_slot_features=extra_slot_features,
        )
        return decode_latent_proposal_for_test(proposal_latent, latent_to_circle_fn=latent_to_center_fn)
