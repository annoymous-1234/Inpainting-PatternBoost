"""Transformer building blocks for the slot-based model backbone."""

from __future__ import annotations

import torch
from torch import nn

from models.features import build_activation


class MultiHeadSelfAttention(nn.Module):
    """Standard full self-attention with optional additive pair bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, pair_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply self-attention to `tokens` of shape `[B, T, d_model]`."""

        batch_size, seq_len, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_logits = torch.matmul(query * self.scale, key.transpose(-2, -1))
        if pair_bias is not None:
            if pair_bias.shape[:3] != (batch_size, seq_len, seq_len):
                raise ValueError("pair_bias must align with the attention sequence length.")
            if pair_bias.shape[-1] == 1:
                bias = pair_bias.permute(0, 3, 1, 2).expand(-1, self.num_heads, -1, -1)
            elif pair_bias.shape[-1] == self.num_heads:
                bias = pair_bias.permute(0, 3, 1, 2)
            else:
                raise ValueError("pair_bias last dimension must be 1 or num_heads.")
            attention_logits = attention_logits + bias.to(
                device=attention_logits.device,
                dtype=attention_logits.dtype,
            )

        attention = torch.softmax(attention_logits, dim=-1)
        attention = self.attn_dropout(attention)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_dropout(self.out_proj(attended))


class FeedForwardMLP(nn.Module):
    """Two-layer Transformer MLP block."""

    def __init__(self, d_model: int, mlp_ratio: float, dropout: float, activation: str) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the MLP block to `[B, T, d_model]` tokens."""

        return self.net(tokens)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional additive pair bias."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float, activation: str) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = FeedForwardMLP(d_model, mlp_ratio, dropout, activation)

    def forward(self, tokens: torch.Tensor, pair_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the pre-norm residual attention + MLP block."""

        tokens = tokens + self.attention(self.norm_attn(tokens), pair_bias=pair_bias)
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens
