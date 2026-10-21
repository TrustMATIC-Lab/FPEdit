
import argparse
from pathlib import Path

import pyiqa
import torch

from metrics.common import image_pairs, image_tensor, resolve_device, write_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    metric = pyiqa.create_metric("topiq_fr", device=device).eval()
    values = []
    with torch.no_grad():
        for source, edited in image_pairs(args.source_dir, args.edited_dir):
            reference = image_tensor(source).unsqueeze(0).to(device)
            distorted = image_tensor(edited).unsqueeze(0).to(device)
            values.append(float(metric(distorted, reference).item()))
    write_result(args.output, "TOPIQ", values, model="topiq_fr")


if __name__ == "__main__":
    main()
