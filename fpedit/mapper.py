
from typing import Optional

import torch
from torch import nn

from fpedit.attention import CrossAttention, SelfAttention


class LevelBranch(nn.Module):
    def __init__(self, options):
        super().__init__()
        kwargs = dict(
            hidden_dim=512,
            qk_out_dim=options.qk_out_dim,
            v_out_dim=options.v_out_dim,
            num_heads=options.num_self_attn_heads,
            dropout=options.mapper_dropout,
            attention_dropout=options.self_attention_dropout,
        )
        self.num_blocks = options.num_attention_blocks
        self.w_self_attention = nn.ModuleList(
            [SelfAttention(**kwargs) for _ in range(options.num_self_attn_layers)]
        )
        self.text_self_attention = nn.ModuleList(
            [SelfAttention(**kwargs) for _ in range(options.num_self_attn_layers)]
        )
        self.cross_attention = nn.ModuleList(
            [
                CrossAttention(
                    kv_dim=512,
                    q_dim=512,
                    qk_out_dim=options.qk_out_dim,
                    v_out_dim=options.v_out_dim,
                    num_heads=options.num_cross_attn_heads,
                    dropout=options.mapper_dropout,
                    attention_dropout=options.cross_attention_dropout,
                )
                for _ in range(options.num_cross_attn_layers)
            ]
        )

    def forward(self, latent, text, text_mask):
        latent_features = latent
        text_features = text
        for _ in range(self.num_blocks):
            for layer in self.w_self_attention:
                latent_features = layer(latent_features)
            for layer in self.text_self_attention:
                text_features = layer(text_features, text_mask)
        fused = latent_features
        for _ in range(self.num_blocks):
            for layer in self.cross_attention:
                fused = layer(text_features, fused, text_mask)
        return fused


class FPEditMapper(nn.Module):
    """Map W+ and contextual CLIP tokens to a three-level latent residual."""

    def __init__(self, options):
        super().__init__()
        self.num_ws = options.num_ws
        if options.self_residual_coefficient != 1.0:
            raise ValueError("SelfAttention residual coefficient must be 1")
        if options.cross_residual_coefficient != 1.0:
            raise ValueError("CrossAttention residual coefficient must be 1")
        if not options.no_coarse_mapper:
            self.coarse_branch = LevelBranch(options)
        if not options.no_medium_mapper:
            self.medium_branch = LevelBranch(options)
        if not options.no_fine_mapper:
            self.fine_branch = LevelBranch(options)

    @staticmethod
    def _text_batch(text, mask, batch_size):
        if text.shape[0] == 1 and batch_size > 1:
            text = text.expand(batch_size, -1, -1)
            mask = mask.expand(batch_size, -1)
        if text.shape[:2] != mask.shape or text.shape[0] != batch_size:
            raise ValueError("Text features, mask, and latent batch do not match")
        return text, mask.bool()

    def forward(self, w_plus, text_features, text_mask):
        if w_plus.ndim != 3 or w_plus.shape[1:] != (self.num_ws, 512):
            raise ValueError(
                f"Expected W+ shape (batch,{self.num_ws},512), got {tuple(w_plus.shape)}"
            )
        text_features, text_mask = self._text_batch(
            text_features, text_mask, w_plus.shape[0]
        )
        boundaries = ((0, 4), (4, 8), (8, self.num_ws))
        names = ("coarse", "medium", "fine")
        outputs = []
        for name, (start, end) in zip(names, boundaries):
            level = w_plus[:, start:end]
            branch: Optional[nn.Module] = getattr(self, f"{name}_branch", None)
            outputs.append(torch.zeros_like(level) if branch is None else branch(level, text_features, text_mask))
        return torch.cat(outputs, dim=1)
