from __future__ import annotations

import torch
from torch import Tensor, nn


class StateScoringHead(nn.Module):
    """Score slot states over ``[MASK]`` and all row values.

    The head emits one logit for ``[MASK]`` and one logit per candidate row for
    each slot token. It mutates only its own parameters during training and
    assumes token order follows the column-pair slot convention.
    """

    def __init__(
        self,
        n: int,
        d_model: int,
        score_dim: int | None = None,
    ) -> None:
        """Initialize the unified state-scoring head.

        ``n`` is the grid size, ``d_model`` is the slot hidden width, and
        ``score_dim`` optionally changes the projected dot-product width. The
        constructor returns ``None`` and mutates only the new module.
        """
        super().__init__()
        self.n = n
        self.head_type = "dot"
        self.score_dim = d_model if score_dim is None else score_dim
        self.mask_head = nn.Linear(d_model, 1)
        self.token_proj = nn.Linear(d_model, self.score_dim)
        self.candidate_proj = nn.Linear(d_model, self.score_dim)

        token_ids = torch.arange(2 * n, dtype=torch.long)
        self.register_buffer("token_to_column", token_ids // 2)

    def forward(
        self,
        slot_hidden: Tensor,
        candidate_point_embeds: Tensor,
    ) -> Tensor:
        """Return unified state logits for every slot token in the batch.

        ``slot_hidden`` is a floating tensor of shape ``[B, 2n, d_model]`` and
        ``candidate_point_embeds`` is a floating tensor of shape ``[n, n,
        d_model]`` with dimensions ``[column, row, feature]``. The method
        returns a floating tensor of shape ``[B, 2n, n + 1]`` on the same
        device, does not mutate the inputs, and assumes the candidate embeddings
        are aligned with the slot-token column mapping.
        """
        if slot_hidden.ndim != 3 or slot_hidden.shape[1] != 2 * self.n:
            raise ValueError(f"slot_hidden must have shape [B, {2 * self.n}, d_model]")
        if candidate_point_embeds.shape[:2] != (self.n, self.n):
            raise ValueError(f"candidate_point_embeds must have shape [{self.n}, {self.n}, d_model]")

        mask_logits = self.mask_head(slot_hidden)
        token_scores = self.token_proj(slot_hidden)
        candidate_scores = self.candidate_proj(candidate_point_embeds)
        candidate_scores_by_token = candidate_scores[self.token_to_column]
        y_logits = torch.einsum("btd,tyd->bty", token_scores, candidate_scores_by_token)
        return torch.cat((mask_logits, y_logits), dim=-1)


class StateClassifierHead(nn.Module):
    """Classify each slot token directly over ``[MASK]`` and all row values.

    The head maps each slot hidden state to ``n + 1`` logits without scoring
    against candidate point embeddings. It mutates only its own parameters
    during training and assumes each slot token already encodes its column and
    pair identity.
    """

    def __init__(
        self,
        n: int,
        d_model: int,
        score_dim: int | None = None,
    ) -> None:
        """Initialize the direct slot-classification head.

        ``n`` is the grid size, ``d_model`` is the slot hidden width, and
        ``score_dim`` optionally inserts a hidden classifier layer of that
        width before the final ``n + 1`` logits. The constructor returns
        ``None`` and mutates only the new module.
        """
        super().__init__()
        self.n = n
        self.head_type = "classifier"
        self.score_dim = score_dim
        if score_dim is None:
            self.classifier = nn.Linear(d_model, n + 1)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, score_dim),
                nn.GELU(),
                nn.Linear(score_dim, n + 1),
            )

    def forward(
        self,
        slot_hidden: Tensor,
        candidate_point_embeds: Tensor,
    ) -> Tensor:
        """Return direct classification logits for every slot token.

        ``slot_hidden`` is a floating tensor of shape ``[B, 2n, d_model]`` and
        ``candidate_point_embeds`` is a floating tensor of shape ``[n, n,
        d_model]`` on the same device, though the classifier head does not use
        the candidate values directly. The method returns a floating tensor of
        shape ``[B, 2n, n + 1]``, does not mutate the inputs, and assumes class
        index ``0`` corresponds to ``[MASK]`` while indices ``1..n`` correspond
        to row ids ``0..n-1``.
        """
        if slot_hidden.ndim != 3 or slot_hidden.shape[1] != 2 * self.n:
            raise ValueError(f"slot_hidden must have shape [B, {2 * self.n}, d_model]")
        if candidate_point_embeds.shape[:2] != (self.n, self.n):
            raise ValueError(f"candidate_point_embeds must have shape [{self.n}, {self.n}, d_model]")
        return self.classifier(slot_hidden)


class ValueHead(nn.Module):
    """Predict a scalar value from the global token hidden state.

    The module is optional and lightweight. It mutates only its own parameters
    during training and assumes the input width matches ``d_model``.
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        """Initialize the value head MLP.

        ``d_model`` is the global hidden width and ``hidden_dim`` optionally
        overrides the intermediate width. The constructor returns ``None`` and
        mutates only the new module.
        """
        super().__init__()
        hidden = d_model if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_hidden: Tensor) -> Tensor:
        """Return one scalar value prediction per batch item.

        ``global_hidden`` is a floating tensor of shape ``[B, d_model]``. The
        method returns a floating tensor of shape ``[B]`` on the same device,
        does not mutate the input tensor, and assumes the last dimension matches
        the configured input width.
        """
        return self.net(global_hidden).squeeze(-1)
