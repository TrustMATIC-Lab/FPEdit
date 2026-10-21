
from typing import Optional

import torch
from torch import nn

from fpedit.stylegan2.model import EqualLinear


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        kv_dim: int = 512,
        q_dim: int = 512,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        qk_out_dim = q_dim if qk_out_dim is None else qk_out_dim
        v_out_dim = qk_out_dim if v_out_dim is None else v_out_dim
        output_dim = v_out_dim if output_dim is None else output_dim
        if qk_out_dim % num_heads or v_out_dim % num_heads:
            raise ValueError("Attention dimensions must be divisible by num_heads")
        self.num_heads = num_heads
        self.qk_head_dim = qk_out_dim // num_heads
        self.v_head_dim = v_out_dim // num_heads
        self.k = nn.Linear(kv_dim, qk_out_dim)
        self.q = nn.Linear(q_dim, qk_out_dim)
        self.v = nn.Linear(kv_dim, v_out_dim)
        self.projection = nn.Linear(v_out_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.qk_head_dim ** -0.5

    def _split_heads(self, tensor: torch.Tensor, head_dim: int) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads, head_dim).transpose(1, 2)

    def forward(
        self,
        inputs_kv: torch.Tensor,
        inputs_q: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        keys = self._split_heads(self.k(inputs_kv), self.qk_head_dim)
        queries = self._split_heads(self.q(inputs_q), self.qk_head_dim)
        values = self._split_heads(self.v(inputs_kv), self.v_head_dim)
        logits = (queries @ keys.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            mask = attention_mask.to(device=logits.device, dtype=torch.bool)
            if mask.ndim != 2 or mask.shape != inputs_kv.shape[:2]:
                raise ValueError("attention_mask must have shape (batch, key_length)")
            logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
        probabilities = self.dropout(logits.softmax(dim=-1))
        weighted = probabilities @ values
        weighted = weighted.transpose(1, 2).reshape(
            inputs_q.shape[0], inputs_q.shape[1], -1
        )
        return self.projection(weighted)


class SelfAttention(nn.Module):
    """Pre-LN self-attention with standard unit double residuals."""

    def __init__(
        self,
        hidden_dim: int = 512,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.qkv_layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(
            hidden_dim, hidden_dim, qk_out_dim, v_out_dim,
            hidden_dim, num_heads, attention_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = EqualLinear(
            hidden_dim, hidden_dim, lr_mul=0.01, activation="fused_lrelu"
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized = self.layer_norm(x)
        hidden = x + self.dropout(
            self.attention(normalized, normalized, attention_mask)
        )
        return hidden + self.mlp(self.qkv_layer_norm(hidden))


class CrossAttention(nn.Module):
    """Text-to-latent attention with a fixed unit query residual."""

    def __init__(
        self,
        kv_dim: int = 512,
        q_dim: int = 512,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.kv_layer_norm = nn.LayerNorm(kv_dim)
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.qkv_layer_norm = nn.LayerNorm(q_dim)
        self.attention = MultiHeadAttention(
            kv_dim, q_dim, qk_out_dim, v_out_dim,
            q_dim, num_heads, attention_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp = EqualLinear(q_dim, q_dim, lr_mul=0.01, activation="fused_lrelu")

    def forward(
        self,
        text: torch.Tensor,
        query: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.attention(
            self.kv_layer_norm(text),
            self.q_layer_norm(query),
            text_mask,
        )
        fused = query + self.dropout(attended)
        return self.mlp(self.qkv_layer_norm(fused))
