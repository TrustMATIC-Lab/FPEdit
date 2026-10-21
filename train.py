
import json
import random
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fpedit.clip_tokens import encode_clip_tokens
from fpedit.data import LatentDataset
from fpedit.losses import build_preservation_loss
from fpedit.model import FPEditModel
from fpedit.options import finish_options, train_parser
from fpedit.ranger import Ranger


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clip_loss(clip_model, edited, token_ids, stylegan_size):
    upsampled = F.interpolate(edited, scale_factor=7, mode="nearest")
    image = F.avg_pool2d(upsampled, kernel_size=stylegan_size // 32)
    return (1.0 - clip_model(image, token_ids)[0] / 100.0).mean()


def save_checkpoint(model, options, step, path):
    config = {
        key: value for key, value in vars(options).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    torch.save(
        {
            "mapper_state_dict": model.mapper.state_dict(),
            "config": config,
            "completed_steps": step,
            "description": options.description,
        },
        path,
    )


@torch.no_grad()
def validate(model, loader, text_features, text_mask, token_ids, loss_id, options, device):
    model.mapper.eval()
    values = []
    for index, w_plus in enumerate(loader):
        if index >= options.validation_batches:
            break
        w_plus = w_plus.to(device)
        source, _ = model.decoder([w_plus], input_is_latent=True, randomize_noise=False)
        delta = model.mapper(w_plus, text_features, text_mask)
        edited, edited_latent, _ = model.decoder(
            [w_plus + options.edit_strength * delta],
            input_is_latent=True,
            return_latents=True,
            randomize_noise=False,
        )
        loss_preservation = loss_id(edited, source)
        loss_clip = clip_loss(model.clip_model, edited, token_ids, options.stylegan_size)
        loss_latent = F.mse_loss(edited_latent, w_plus)
        total = (
            options.id_lambda * loss_preservation
            + options.clip_lambda * loss_clip
            + options.latent_l2_lambda * loss_latent
        )
        values.append(
            [total.item(), loss_preservation.item(), loss_clip.item(), loss_latent.item()]
        )
    model.mapper.train()
    mean = np.asarray(values, dtype=np.float64).mean(axis=0)
    return dict(zip(("loss", "loss_id", "loss_clip", "loss_latent"), mean.tolist()))


def main():
    options = finish_options(train_parser().parse_args())
    options.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(options.seed)
    device = torch.device(options.device)

    model = FPEditModel(options).to(device)
    model.freeze_decoder()
    model.clip_model, _ = clip.load(options.clip_model, device=device)
    model.clip_model.eval().requires_grad_(False)

    preservation_loss = build_preservation_loss(
        options.domain, options.identity_ckpt
    ).to(device).eval()
    preservation_loss.requires_grad_(False)

    token_ids = clip.tokenize(options.description).to(device)
    text_features, text_mask = encode_clip_tokens(model.clip_model, token_ids)
    text_features = text_features.detach()
    text_mask = text_mask.detach()

    train_data = LatentDataset(options.train_latents, options.num_ws)
    val_data = LatentDataset(options.val_latents, options.num_ws)
    train_loader = DataLoader(
        train_data, batch_size=options.batch_size, shuffle=True,
        num_workers=options.workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=options.val_batch_size, shuffle=False,
        num_workers=options.workers, drop_last=True,
    )
    parameters = model.mapper.parameters()
    optimizer = (
        torch.optim.Adam(parameters, lr=options.learning_rate)
        if options.optimizer == "adam"
        else Ranger(parameters, lr=options.learning_rate)
    )
    writer = SummaryWriter(options.output_dir / "logs")
    history = {"train": [], "validation": []}
    step = 0
    progress = tqdm(total=options.max_steps, desc=options.description)

    model.mapper.train()
    while step < options.max_steps:
        for w_plus in train_loader:
            if step >= options.max_steps:
                break
            w_plus = w_plus.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                source, _ = model.decoder(
                    [w_plus], input_is_latent=True, randomize_noise=False
                )
            delta = model.mapper(w_plus, text_features, text_mask)
            edited, edited_latent, _ = model.decoder(
                [w_plus + options.edit_strength * delta],
                input_is_latent=True,
                return_latents=True,
                randomize_noise=False,
            )
            loss_id = preservation_loss(edited, source)
            loss_clip_value = clip_loss(
                model.clip_model, edited, token_ids, options.stylegan_size
            )
            loss_latent = F.mse_loss(edited_latent, w_plus)
            loss = (
                options.id_lambda * loss_id
                + options.clip_lambda * loss_clip_value
                + options.latent_l2_lambda * loss_latent
            )
            loss.backward()
            optimizer.step()
            step += 1
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "loss_id": float(loss_id.detach()),
                "loss_clip": float(loss_clip_value.detach()),
                "loss_latent": float(loss_latent.detach()),
            }
            history["train"].append(record)
            for name, value in record.items():
                if name != "step":
                    writer.add_scalar(f"train/{name}", value, step)

            if step % options.image_interval == 0:
                torchvision.utils.save_image(
                    torch.cat((source.detach(), edited.detach())),
                    options.output_dir / f"images_{step:06d}.png",
                    normalize=True,
                    value_range=(-1, 1),
                    nrow=options.batch_size,
                )
            if step % options.val_interval == 0 or step == options.max_steps:
                result = validate(
                    model, val_loader, text_features, text_mask, token_ids,
                    preservation_loss, options, device,
                )
                result["step"] = step
                history["validation"].append(result)
                for name, value in result.items():
                    if name != "step":
                        writer.add_scalar(f"validation/{name}", value, step)
            if step % options.save_interval == 0 or step == options.max_steps:
                save_checkpoint(
                    model, options, step,
                    options.output_dir / f"iteration_{step}.pt",
                )
            progress.update(1)

    progress.close()
    writer.close()
    with (options.output_dir / "loss_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with (options.output_dir / "options.json").open("w", encoding="utf-8") as file:
        json.dump({key: str(value) for key, value in vars(options).items()}, file, indent=2)


if __name__ == "__main__":
    main()
