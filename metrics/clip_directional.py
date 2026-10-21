
import argparse
from pathlib import Path

import clip
import torch

from metrics.common import image_pairs, load_pil, prompt_values, resolve_device, write_result


def normalize(feature):
    return feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ffhq", "cat", "church"), required=True)
    parser.add_argument("--prompt-key", required=True)
    parser.add_argument("--prompts-yaml", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    model.eval().requires_grad_(False)
    prompts = prompt_values(args.prompts_yaml, args.dataset, args.prompt_key)
    tokens = clip.tokenize([prompts["clip_target"], prompts["neutral"]]).to(device)
    with torch.no_grad():
        text = normalize(model.encode_text(tokens).float())
    text_direction = text[0] - text[1]
    text_direction = text_direction / text_direction.norm().clamp_min(1e-12)
    values = []
    with torch.no_grad():
        for source, edited in image_pairs(args.source_dir, args.edited_dir):
            images = torch.stack(
                [preprocess(load_pil(source)), preprocess(load_pil(edited))]
            ).to(device)
            features = normalize(model.encode_image(images).float())
            direction = features[1] - features[0]
            direction = direction / direction.norm().clamp_min(1e-12)
            values.append(float(torch.dot(direction, text_direction).item()))
    write_result(
        args.output, "CLIP Directional Similarity", values,
        target_prompt=prompts["clip_target"], neutral_prompt=prompts["neutral"],
        model=args.clip_model,
    )


if __name__ == "__main__":
    main()
