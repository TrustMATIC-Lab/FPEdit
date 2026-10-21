
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50

from fpedit.recognition.model_irse import Backbone


class FaceIdentityLoss(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.encoder = Backbone(112, 50, "ir_se", 0.6)
        self.encoder.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        self.encoder.eval().requires_grad_(False)

    def features(self, image):
        image = F.adaptive_avg_pool2d(image, (256, 256))
        image = image[:, :, 35:223, 32:220]
        return self.encoder(F.adaptive_avg_pool2d(image, (112, 112)))

    def forward(self, edited, source):
        source_features = self.features(source).detach()
        edited_features = self.features(edited)
        return (1.0 - (source_features * edited_features).sum(dim=1)).mean()


class MocoPreservationLoss(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        network = resnet50(weights=None)
        source = torch.load(checkpoint, map_location="cpu")["state_dict"]
        state = {
            key[len("module.encoder_q."):]: value
            for key, value in source.items()
            if key.startswith("module.encoder_q.")
            and not key.startswith("module.encoder_q.fc")
        }
        network.load_state_dict(state, strict=False)
        self.encoder = nn.Sequential(*list(network.children())[:-1])
        self.encoder.eval().requires_grad_(False)

    def features(self, image):
        image = F.interpolate(image, size=224)
        return F.normalize(self.encoder(image).flatten(1), dim=1)

    def forward(self, edited, source):
        source_features = self.features(source).detach()
        edited_features = self.features(edited)
        return (1.0 - (source_features * edited_features).sum(dim=1)).mean()


def build_preservation_loss(domain, checkpoint):
    return FaceIdentityLoss(checkpoint) if domain == "ffhq" else MocoPreservationLoss(checkpoint)
