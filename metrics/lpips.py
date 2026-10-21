
import argparse
from pathlib import Path

import lpips
import torch

from metrics.common import image_pairs, image_tensor, resolve_device, write_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--net", choices=("alex", "vgg", "squeeze"), default="alex")
    args = parser.parse_args()
    device = resolve_device(args.device)
    metric = lpips.LPIPS(net=args.net, spatial=False, verbose=False).to(device).eval()
    values = []
    with torch.no_grad():
        for source, edited in image_pairs(args.source_dir, args.edited_dir):
            x = image_tensor(source, minus_one=True).unsqueeze(0).to(device)
            y = image_tensor(edited, minus_one=True).unsqueeze(0).to(device)
            values.append(float(metric(x, y, normalize=False).item()))
    write_result(args.output, "LPIPS", values, network=args.net)


if __name__ == "__main__":
    main()
