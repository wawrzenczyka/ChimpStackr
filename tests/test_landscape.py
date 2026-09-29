"""Behavioral checks for short landscape stacks and their alignment."""

import cv2
import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

from src.algorithms import Algorithm
from src.algorithms.API import LaplacianPyramid
from src.config import AlgorithmConfig
from src.algorithms.stacking_algorithms.landscape import fuse


def _focus_pair():
    rng = np.random.default_rng(42)
    base = rng.integers(35, 220, (180, 240, 3)).astype(np.float32) + 0.375
    soft = cv2.GaussianBlur(base, (0, 0), 3)
    left = base.copy()
    left[:, 120:] = soft[:, 120:]
    right = base.copy()
    right[:, :120] = soft[:, :120]
    return base, left, right


@pytest.mark.parametrize("blend", [False, True])
def test_landscape_recovers_sharp_regions_and_keeps_float_precision(blend):
    base, left, right = _focus_pair()
    result, labels, confidence = fuse([left, right], radius=5, blend=blend)
    assert result.dtype == np.float32
    assert labels.dtype == np.uint8
    assert confidence.shape == base.shape[:2]
    assert np.isfinite(result).all()
    assert (labels[20:-20, 20:100] == 0).mean() > 0.9
    assert (labels[20:-20, 140:-20] == 1).mean() > 0.9
    np.testing.assert_allclose(result[30:70, 30:70], base[30:70, 30:70], atol=1e-4)
    np.testing.assert_allclose(result[30:70, 170:210], base[30:70, 170:210], atol=1e-4)


def test_landscape_never_selects_invalid_warp_border():
    _, left, right = _focus_pair()
    left[:, :30] = 0
    valid_left = np.ones(left.shape[:2], bool)
    valid_left[:, :30] = False
    result, labels, _ = fuse([left, right], [valid_left, np.ones_like(valid_left)])
    assert np.all(labels[:, :30] == 1)
    np.testing.assert_array_equal(result[:, :30], right[:, :30])


def test_landscape_source_order_is_stable_in_confident_regions():
    _, left, right = _focus_pair()
    first, _, _ = fuse([left, right], radius=5)
    reverse, _, _ = fuse([right, left], radius=5)
    np.testing.assert_allclose(first[25:-25, 25:95], reverse[25:-25, 25:95])
    np.testing.assert_allclose(first[25:-25, 145:-25], reverse[25:-25, 145:-25])


def test_landscape_cancellation_leaves_no_output():
    algo = LaplacianPyramid(AlgorithmConfig(stacking_method="landscape"))
    algo.update_image_paths([
        "tests/low_res_images/DSC_0356.jpg",
        "tests/low_res_images/DSC_0358.jpg",
    ])
    algo.stack_images(progress_callback=lambda *_: algo.cancel())
    assert algo.output_image is None


def test_auto_alignment_uses_landscape_for_landscape_methods():
    for method in ("landscape", "landscape_blend"):
        algo = LaplacianPyramid(AlgorithmConfig(stacking_method=method, alignment_mode="auto"))
        assert algo._resolve_alignment_mode() == "landscape"


def test_landscape_alignment_reduces_known_translation():
    rng = np.random.default_rng(1)
    reference = rng.integers(0, 256, (320, 420, 3), dtype=np.uint8).astype(np.float32)
    shifted = cv2.warpAffine(reference, np.float32([[1, 0, 7], [0, 1, -5]]),
                             (420, 320), borderMode=cv2.BORDER_CONSTANT)
    algorithm = Algorithm()
    aligned = algorithm.align_image_pair(reference, shifted, alignment_mode="landscape")
    before = np.mean(np.abs(reference[30:-30, 30:-30] - shifted[30:-30, 30:-30]))
    after = np.mean(np.abs(reference[30:-30, 30:-30] - aligned[30:-30, 30:-30]))
    assert after < before * 0.2
    assert algorithm.last_alignment_mask.shape == reference.shape[:2]


def test_cli_runs_both_landscape_methods_with_separate_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "comparison.tif"
    completed = subprocess.run(
        [sys.executable, "-m", "src.cli", "--input",
         "tests/low_res_images/DSC_0356.jpg", "tests/low_res_images/DSC_0358.jpg",
         "--output", str(output), "--method", "landscape,landscape_blend",
         "--align", "--bit-depth", "16"],
        cwd=root, capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("mode=landscape") == 2
    for method in ("landscape", "landscape_blend"):
        path = tmp_path / f"comparison_{method}.tif"
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image is not None
        assert image.dtype == np.uint16
