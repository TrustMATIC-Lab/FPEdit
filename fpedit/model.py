
from types import SimpleNamespace

import torch
from torch import nn

from fpedit.mapper import FPEditMapper
from fpedit.stylegan2.model import Generator


def mapper_options_from_checkpoint(checkpoint, fallback):
    values = vars(fallback).copy()
    values.update(checkpoint.get("config", {}))
    values["num_ws"] = fallback.num_ws
    values["stylegan_size"] = fallback.stylegan_size
    return SimpleNamespace(**values)


class FPEditModel(nn.Module):
    def __init__(self, options, mapper_checkpoint=None):
        super().__init__()
        checkpoint = mapper_checkpoint or {}
        mapper_options = mapper_options_from_checkpoint(checkpoint, options)
        self.mapper = FPEditMapper(mapper_options)
        self.decoder = Generator(options.stylegan_size, 512, 8)
        stylegan = torch.load(options.stylegan_ckpt, map_location="cpu")
        generator_state = stylegan["g_ema"] if "g_ema" in stylegan else stylegan
        self.decoder.load_state_dict(generator_state, strict=False)
        if checkpoint:
            mapper_state = checkpoint.get("mapper_state_dict", checkpoint)
            self.mapper.load_state_dict(mapper_state, strict=True)

    def freeze_decoder(self):
        self.decoder.eval()
        self.decoder.requires_grad_(False)
