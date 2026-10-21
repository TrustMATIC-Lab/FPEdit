
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
import torchvision
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from fpedit.clip_tokens import encode_clip_tokens
from fpedit.data import ImageDataset
from fpedit.encoders import Encoder4Editing
from fpedit.model import FPEditModel
from fpedit.options import finish_options


def prefixed_state(checkpoint, prefix):
    state = checkpoint.get("state_dict", checkpoint)
    key_prefix = prefix + "."
    return {
        key[len(key_prefix):]: value
        for key, value in state.items()
        if key.startswith(key_prefix)
    }


def prompt_text(path: Path, domain: str, prompt_key: str) -> str:
    with path.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file)
    return registry[domain]["prompts"][prompt_key]["model_prompt"]


def run_inference(options):
    options = finish_options(options)
    device = torch.device(options.device)
    checkpoint = torch.load(options.mapper_ckpt, map_location="cpu")
    model = FPEditModel(options, checkpoint).to(device).eval()
    model.freeze_decoder()
    clip_model, _ = clip.load(options.clip_model, device=device)
    clip_model.eval().requires_grad_(False)

    description = prompt_text(options.prompts_yaml, options.domain, options.prompt_key)
    token_ids = clip.tokenize(description).to(device)
    text_features, text_mask = encode_clip_tokens(clip_model, token_ids)

    e4e_checkpoint = torch.load(options.e4e_ckpt, map_location="cpu")
    encoder = Encoder4Editing(50, options.stylegan_size, "ir_se")
    encoder.load_state_dict(prefixed_state(e4e_checkpoint, "encoder"), strict=True)
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    latent_average = e4e_checkpoint["latent_avg"].to(device)

    dataset = ImageDataset(options.input_dir)
    loader = DataLoader(
        dataset, batch_size=options.batch_size, shuffle=False,
        num_workers=options.workers, drop_last=False,
    )
    options.output_dir.mkdir(parents=True, exist_ok=True)
    if options.reconstruction_dir:
        options.reconstruction_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, names in tqdm(loader, desc=f"{options.domain}/{options.prompt_key}"):
            images = images.to(device)
            w_plus = encoder(images)
            if latent_average.ndim == 1:
                latent_average = latent_average.unsqueeze(0).unsqueeze(0)
            elif latent_average.ndim == 2:
                latent_average = latent_average.unsqueeze(0)
            w_plus = w_plus + latent_average
            delta = model.mapper(w_plus, text_features, text_mask)
            reconstruction, _ = model.decoder(
                [w_plus], input_is_latent=True, randomize_noise=False
            )
            edited, _ = model.decoder(
                [w_plus + options.edit_strength * delta],
                input_is_latent=True,
                randomize_noise=False,
            )
            reconstruction = F.interpolate(
                reconstruction, size=(256, 256), mode="bilinear", align_corners=False
            )
            edited = F.interpolate(
                edited, size=(256, 256), mode="bilinear", align_corners=False
            )
            for index, name in enumerate(names):
                torchvision.utils.save_image(
                    edited[index], options.output_dir / name,
                    normalize=True, value_range=(-1, 1),
                )
                if options.reconstruction_dir:
                    torchvision.utils.save_image(
                        reconstruction[index], options.reconstruction_dir / name,
                        normalize=True, value_range=(-1, 1),
                    )
