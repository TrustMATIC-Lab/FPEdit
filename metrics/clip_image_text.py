
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
    prompt = prompt_values(args.prompts_yaml, args.dataset, args.prompt_key)["clip_target"]
    with torch.no_grad():
        text = normalize(model.encode_text(clip.tokenize([prompt]).to(device)).float())[0]
    values = []
    with torch.no_grad():
        for _, edited in image_pairs(args.source_dir, args.edited_dir):
            image = preprocess(load_pil(edited)).unsqueeze(0).to(device)
            feature = normalize(model.encode_image(image).float())[0]
            values.append(float(torch.dot(feature, text).item()))
    write_result(
        args.output, "CLIP Image-Text Similarity", values,
        target_prompt=prompt, model=args.clip_model,
    )


if __name__ == "__main__":
    main()
