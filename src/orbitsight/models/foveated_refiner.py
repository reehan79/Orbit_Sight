from __future__ import annotations

import torch
from torch import nn

from orbitsight.features.candidate_features import FEATURE_NAMES
from orbitsight.features.event_patch import PATCH_CHANNELS, PATCH_SIZE

NUM_CANDIDATE_FEATURES = len(FEATURE_NAMES)
FLATTEN_DIM = 32 * 4 * 4  # 512 after three stride-2 convs on 32x32
HIDDEN = 64
BBOX_DIM = 4


class TinyFoveatedRefiner(nn.Module):
    """Fixed tiny local neural refiner. Architecture is frozen — do not alter."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(PATCH_CHANNELS, 12, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(FLATTEN_DIM + NUM_CANDIDATE_FEATURES, HIDDEN)
        self.cls_head = nn.Linear(HIDDEN, 1)
        self.bbox_head = nn.Linear(HIDDEN, BBOX_DIM)

    def forward(self, patch: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.conv1(patch))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x, features], dim=1)
        x = torch.relu(self.fc(x))
        return self.cls_head(x).squeeze(-1), self.bbox_head(x)


def parameter_count(model: nn.Module | None = None) -> int:
    net = model if model is not None else TinyFoveatedRefiner()
    return int(sum(p.numel() for p in net.parameters()))
