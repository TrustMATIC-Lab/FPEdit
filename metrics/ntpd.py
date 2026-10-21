
import argparse
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F

from metrics.common import image_tensor, mask_pairs, resolve_device, target_mask, write_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--net", choices=("alex", "vgg", "squeeze"), default="alex")
    args = parser.parse_args()
    device = resolve_device(args.device)
    metric = lpips.LPIPS(net=args.net, spatial=True, verbose=False).to(device).eval()
    values = []
    with torch.no_grad():
        for source, edited, mask_path in mask_pairs(
            args.source_dir, args.edited_dir, args.mask_dir
        ):
            x = image_tensor(source, minus_one=True).unsqueeze(0).to(device)
            y = image_tensor(edited, minus_one=True).unsqueeze(0).to(device)
            change = metric(x, y, normalize=False).float().clamp_min(0.0)
            change = F.interpolate(change, size=(256, 256), mode="bilinear", align_corners=False)[0, 0]
            mask = target_mask(mask_path).to(device)
            values.append(float(change[~mask].mean().item()))
    write_result(args.output, "NTPD", values, mask_convention="white=target")


if __name__ == "__main__":
    main()
