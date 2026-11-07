from __future__ import annotations

from torch import Tensor, nn

from .backbone import TransformerBackbone
from .embeddings import SlotTokenEmbedding
from .heads import StateClassifierHead, StateScoringHead, ValueHead
from .point_features import PointFeatureConfig, load_point_feature_config


class ColumnPairModel(nn.Module):
    """Assemble embeddings, backbone, and heads for the column-pair model.

    The model consumes batched slot-state ids and returns neural outputs only.
    It does not perform proposal selection or mutate environment state.
    """

    def __init__(
        self,
        n: int,
        num_steps: int,
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.0,
        point_feature_config: PointFeatureConfig | None = None,
        use_global_token: bool = True,
        use_value_head: bool = False,
        score_dim: int | None = None,
        state_head_type: str = "dot",
        time_embedding_type: str = "learned",
    ) -> None:
        """Initialize the full column-pair model.

        ``n`` is the grid size and ``num_steps`` sizes the progress embedding table.
        The remaining arguments configure the Transformer, feature projection,
        optional global token, optional value head, and the state-head
        implementation. ``state_head_type`` must be ``"dot"`` for the original
        candidate-scoring head or ``"classifier"`` for direct ``n + 1`` slot
        classification, and ``time_embedding_type`` selects learned or fixed
        sinusoidal progress embeddings. The constructor returns ``None``,
        mutates only the new module, and assumes the requested widths and head
        counts are compatible.
        """
        super().__init__()
        if use_value_head and not use_global_token:
            raise ValueError("use_value_head=True requires use_global_token=True")
        if state_head_type not in {"dot", "classifier"}:
            raise ValueError("state_head_type must be 'dot' or 'classifier'")

        self.n = n
        self.use_global_token = use_global_token
        self.use_value_head = use_value_head
        self.state_head_type = state_head_type

        config = load_point_feature_config() if point_feature_config is None else point_feature_config
        self.token_embedding = SlotTokenEmbedding(
            n=n,
            d_model=d_model,
            num_steps=num_steps,
            point_feature_config=config,
            use_global_token=use_global_token,
            time_embedding_type=time_embedding_type,
        )
        self.backbone = TransformerBackbone(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        if state_head_type == "dot":
            self.state_head = StateScoringHead(n=n, d_model=d_model, score_dim=score_dim)
        else:
            self.state_head = StateClassifierHead(n=n, d_model=d_model, score_dim=score_dim)
        self.value_head = ValueHead(d_model=d_model) if use_value_head else None

    def forward(
        self,
        slot_state_ids: Tensor,
        step_ids: Tensor,
        current_point_counts: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Run the model forward pass without decoding or environment mutation.

        ``slot_state_ids`` is a ``LongTensor`` of shape ``[B, 2n]``,
        ``step_ids`` is a ``LongTensor`` of filled-slot progress ids with shape
        ``[B]``, and ``current_point_counts`` is optional with shape ``[B]`` or
        ``[B, 1]``. The method returns a dictionary containing ``state_logits``
        of shape ``[B, 2n, n + 1]`` and optionally ``value`` of shape ``[B]``.
        It does not mutate the input tensors or environment state and assumes
        the slot ids follow the column-pair encoding.
        """
        embedding_out = self.token_embedding(
            slot_state_ids=slot_state_ids,
            step_ids=step_ids,
            current_point_counts=current_point_counts,
        )
        hidden = self.backbone(embedding_out["token_embeds"])

        if self.use_global_token:
            global_hidden = hidden[:, 0, :]
            slot_hidden = hidden[:, 1:, :]
        else:
            global_hidden = None
            slot_hidden = hidden

        state_logits = self.state_head(slot_hidden, embedding_out["candidate_point_embeds"])
        value = self.value_head(global_hidden) if self.value_head is not None and global_hidden is not None else None
        return {
            "state_logits": state_logits,
            "value": value,
        }
