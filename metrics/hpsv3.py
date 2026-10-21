
import argparse
from pathlib import Path

import torch
from hpsv3 import HPSv3RewardInferencer

from metrics.common import image_pairs, prompt_values, write_result


def scalar_reward(value):
    while isinstance(value, (list, tuple)):
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 2:
            return float(value[0, 0])
        if value.ndim == 1:
            return float(value[0])
        return float(value.item())
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ffhq", "cat", "church"), required=True)
    parser.add_argument("--prompt-key", required=True)
    parser.add_argument("--prompts-yaml", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    prompt = prompt_values(args.prompts_yaml, args.dataset, args.prompt_key)["hpsv3_target"]
    inferencer = HPSv3RewardInferencer(device=args.device)
    values = []
    for _, edited in image_pairs(args.source_dir, args.edited_dir):
        with torch.no_grad():
            reward = inferencer.reward(image_paths=[str(edited)], prompts=[prompt])
        values.append(scalar_reward(reward))
    write_result(args.output, "HPSv3", values, prompt=prompt)


if __name__ == "__main__":
    main()
