"""Shared image pairing, preprocessing, prompts, and JSON output."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_pairs(source_dir: Path, edited_dir: Path):
    sources = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not sources:
        raise ValueError(f"No images found in {source_dir}")
    pairs = []
    for source in sources:
        edited = edited_dir / source.name
        if not edited.is_file():
            raise FileNotFoundError(f"Missing same-name edited image: {edited}")
        pairs.append((source, edited))
    return pairs


def load_pil(path: Path, size: int = 256, nearest: bool = False):
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    with Image.open(path) as image:
        return image.convert("RGB").resize((size, size), resample)


def image_tensor(path: Path, minus_one: bool = False):
    array = np.asarray(load_pil(path), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor * 2.0 - 1.0 if minus_one else tensor


def target_mask(path: Path):
    with Image.open(path) as image:
        gray = np.asarray(
            image.convert("L").resize((256, 256), Image.Resampling.NEAREST)
        )
    return torch.from_numpy(gray >= 128)


def prompt_values(path: Path, dataset: str, prompt_key: str):
    with path.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file)
    dataset_spec = registry[dataset]
    prompt_spec = dataset_spec["prompts"][prompt_key]
    return {
        "model": prompt_spec["model_prompt"],
        "clip_target": prompt_spec["clip_target"],
        "hpsv3_target": prompt_spec["hpsv3_target"],
        "neutral": dataset_spec["neutral_prompt"],
    }


def resolve_device(value: str):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def write_result(output: Path, metric: str, values, **metadata):
    array = np.asarray(values, dtype=np.float64)
    result = {
        "metric": metric,
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "n": int(array.size),
        **metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def mask_pairs(source_dir: Path, edited_dir: Path, mask_dir: Path):
    for source, edited in image_pairs(source_dir, edited_dir):
        mask = mask_dir / source.name
        if not mask.is_file():
            raise FileNotFoundError(f"Missing same-name normalized mask: {mask}")
        yield source, edited, mask
