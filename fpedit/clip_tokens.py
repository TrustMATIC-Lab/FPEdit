
from typing import Tuple

import torch


@torch.no_grad()
def encode_clip_tokens(
    clip_model,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (batch, context_length)")

    dtype = getattr(clip_model, "dtype", clip_model.token_embedding.weight.dtype)
    x = clip_model.token_embedding(token_ids).to(dtype=dtype)
    x = x + clip_model.positional_embedding[: token_ids.shape[1]].to(dtype=dtype)
    x = clip_model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = clip_model.ln_final(x).to(dtype=dtype)
    if getattr(clip_model, "text_projection", None) is not None:
        x = x @ clip_model.text_projection

    eot = token_ids.argmax(dim=-1)
    positions = torch.arange(token_ids.shape[1], device=token_ids.device)
    mask = positions.unsqueeze(0) <= eot.unsqueeze(1)
    length = int(eot.max().item()) + 1
    x = x[:, :length]
    mask = mask[:, :length]
    if torch.any(mask.sum(dim=1) < 2):
        raise ValueError("Every prompt must contain at least SOT and EOT")
    return x.float(), mask.bool()
