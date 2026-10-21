
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50

from fpedit.recognition import Backbone
from metrics.common import image_pairs, image_tensor, resolve_device, write_result


class FaceEncoder(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.model = Backbone(112, 50, "ir_se", 0.6)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)

    def forward(self, image):
        image = image[:, :, 35:223, 32:220]
        return self.model(F.adaptive_avg_pool2d(image, (112, 112)))


class MocoEncoder(nn.Module):
    def __init__(self, weights):
        super().__init__()
        model = resnet50(weights=None)
        source = torch.load(weights, map_location="cpu")["state_dict"]
        state = {
            key[len("module.encoder_q."):]: value
            for key, value in source.items()
            if key.startswith("module.encoder_q.")
            and not key.startswith("module.encoder_q.fc")
        }
        model.load_state_dict(state, strict=False)
        self.model = nn.Sequential(*list(model.children())[:-1])

    def forward(self, image):
        feature = self.model(F.interpolate(image, size=224)).flatten(1)
        return F.normalize(feature, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", choices=("ffhq", "cat", "church"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    encoder = (
        FaceEncoder(args.weights) if args.domain == "ffhq"
        else MocoEncoder(args.weights)
    ).to(device).eval()
    encoder.requires_grad_(False)
    values = []
    with torch.no_grad():
        for source, edited in image_pairs(args.source_dir, args.edited_dir):
            x = image_tensor(source, minus_one=True).unsqueeze(0).to(device)
            y = image_tensor(edited, minus_one=True).unsqueeze(0).to(device)
            values.append(float((encoder(x) * encoder(y)).sum(dim=1).item()))
    write_result(args.output, "IDS", values, domain=args.domain)


if __name__ == "__main__":
    main()
