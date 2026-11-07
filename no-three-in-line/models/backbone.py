from __future__ import annotations

from torch import Tensor, nn


class PreNormTransformerBlock(nn.Module):
    """Run one pre-norm self-attention plus feed-forward residual block.

    The block preserves sequence length and hidden width. It mutates only its
    own parameters during training and assumes the input hidden size matches the
    configured ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize one Transformer encoder block.

        ``d_model`` is the token width, ``num_heads`` is the attention head
        count, ``d_ff`` is the feed-forward hidden width, and ``dropout`` is
        used in attention and FFN sublayers. The constructor returns ``None``
        and mutates only the new module.
        """
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block to a batch of token embeddings.

        ``x`` is a floating tensor of shape ``[B, N, d_model]`` on the module
        device. The method returns a tensor of the same shape and dtype, does
        not mutate the input tensor in place, and assumes ``d_model`` matches
        the layer configuration.
        """
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attn_dropout(attn_output)
        x = x + self.ff(self.ff_norm(x))
        return x


class TransformerBackbone(nn.Module):
    """Stack multiple pre-norm Transformer blocks without changing shape.

    The backbone is generic and sequence-agnostic. It mutates only its own
    parameters during training and assumes every layer uses the same hidden
    width as the input tensor.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the Transformer encoder stack.

        ``num_layers`` controls depth and the remaining arguments are forwarded
        to each block. The constructor returns ``None``, mutates only the new
        module, and assumes the provided dimensions are compatible.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PreNormTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the encoder stack over a batch of token embeddings.

        ``x`` is a floating tensor of shape ``[B, N, d_model]``. The method
        returns a tensor of the same shape and dtype, does not mutate the input
        tensor in place, and assumes ``x`` already uses the configured hidden
        width.
        """
        for layer in self.layers:
            x = layer(x)
        return x
