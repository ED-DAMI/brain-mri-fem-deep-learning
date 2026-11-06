import numpy as np
import torch

from brain_mri_fem.preprocessing import normalize_volume, patient_splits
from brain_mri_fem.segmentation import UNet3D, dice, iou


def test_normalization_has_no_nan_and_zero_background():
    result = normalize_volume(np.array([[[0.0, 1.0], [2.0, 3.0]]]))
    assert np.isfinite(result).all()
    assert result[0, 0, 0] == 0


def test_patient_splits_are_disjoint_and_reproducible():
    splits = patient_splits([f"p{i}" for i in range(10)])
    assert len(set(splits["train"]) & set(splits["val"])) == 0
    assert splits == patient_splits([f"p{i}" for i in range(10)])


def test_metrics_identity_and_disjoint():
    first = np.array([1, 0, 0])
    second = np.array([0, 1, 0])
    assert dice(first, first) == 1
    assert iou(first, second) < 1e-7


def test_unet_forward_shape():
    model = UNet3D(channels=(4, 8))
    output = model(torch.zeros((1, 1, 8, 8, 8)))
    assert output.shape == (1, 2, 8, 8, 8)
