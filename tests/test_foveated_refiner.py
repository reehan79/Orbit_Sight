import numpy as np
import torch

from orbitsight.features.event_patch import PATCH_CHANNELS, PATCH_SIZE, rasterize_event_patch
from orbitsight.models.foveated_refiner import TinyFoveatedRefiner, parameter_count


def test_rasterize_event_patch_shape_and_channels():
    current = np.array(
        [
            [10.0, 10.0, 1.0, 0.0],
            [11.0, 10.0, 0.0, 20_000.0],
        ],
        dtype=np.float64,
    )
    prior = np.array([[10.0, 10.0, 1.0, -10_000.0]], dtype=np.float64)
    patch = rasterize_event_patch(current, prior, 10.0, 10.0, cell=4.0, start_us=0, end_us=40_000)
    assert patch.shape == (PATCH_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    assert patch.dtype == np.float16
    assert float(patch[0].sum()) > 0.0
    assert float(patch[1].sum()) > 0.0
    assert float(patch[3].sum()) > 0.0


def test_tiny_foveated_refiner_param_count_and_forward():
    model = TinyFoveatedRefiner()
    n = parameter_count(model)
    assert n > 0
    patch = torch.zeros(2, 4, 32, 32)
    feats = torch.zeros(2, 15)
    cls, bbox = model(patch, feats)
    assert cls.shape == (2,)
    assert bbox.shape == (2, 4)
