"""Behavioral checks for short landscape stacks and their alignment."""

import cv2
import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

from src.algorithms import Algorithm
from src.algorithms import learned_alignment
from src.algorithms.API import LaplacianPyramid
from src.config import AlgorithmConfig
from src.algorithms.stacking_algorithms.landscape import fuse
from src.algorithms.stacking_algorithms.region_fusion import fuse_near_far, fuse_regions
from src.algorithms.stacking_algorithms import depth_guided


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


@pytest.mark.parametrize("region_fuse", [fuse_near_far, fuse_regions])
def test_region_fusion_recovers_two_focus_planes(region_fuse):
    base, near, far = _focus_pair()
    result, labels, confidence = region_fuse([near, far], radius=5)
    assert result.dtype == np.float32
    assert labels[30:-30, 30:90].mean() < 0.1
    assert labels[30:-30, 150:210].mean() > 0.9
    np.testing.assert_allclose(result[40:70, 30:80], base[40:70, 30:80])
    np.testing.assert_allclose(result[40:70, 160:210], base[40:70, 160:210])
    assert np.isfinite(confidence).all()


def test_landscape_regions_selects_multiple_near_and_far_sources():
    rng = np.random.default_rng(8)
    base = rng.integers(30, 220, (180, 240, 3)).astype(np.float32) + 0.125
    soft = cv2.GaussianBlur(base, (0, 0), 4)
    images = []
    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
        image = soft.copy()
        ys = slice(row * 90, (row + 1) * 90)
        xs = slice(col * 120, (col + 1) * 120)
        image[ys, xs] = base[ys, xs]
        images.append(image)
    result, labels, _ = fuse_regions(images, radius=5)
    for index, (row, col) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        ys = slice(row * 90 + 25, (row + 1) * 90 - 25)
        xs = slice(col * 120 + 25, (col + 1) * 120 - 25)
        assert (labels[ys, xs] == index).mean() > 0.9
        np.testing.assert_allclose(result[ys, xs], base[ys, xs])


def test_near_far_cut_obeys_warp_validity_and_pair_limit():
    _, near, far = _focus_pair()
    near_valid = np.ones(near.shape[:2], bool)
    near_valid[:, :20] = False
    result, labels, _ = fuse_near_far([near, far], [near_valid, np.ones_like(near_valid)])
    assert np.all(labels[:, :20] == 1)
    np.testing.assert_array_equal(result[:, :20], far[:, :20])
    with pytest.raises(ValueError, match="exactly two"):
        fuse_near_far([near, far, near])


def test_depth_guided_selects_near_object_and_far_scene_without_source_order(monkeypatch):
    rng = np.random.default_rng(29)
    base = rng.integers(30, 220, (180, 240, 3)).astype(np.float32) + 0.25
    soft = cv2.GaussianBlur(base, (0, 0), 4)
    object_area = np.s_[35:155, 55:185]
    far = base.copy()
    far[object_area] = soft[object_area]
    near = soft.copy()
    near[object_area] = base[object_area]
    far_depth = np.zeros(base.shape[:2], np.float32)
    far_depth[145:] = 2
    near_depth = np.zeros_like(far_depth)
    near_depth[object_area] = 2

    for images, maps, near_index in (([far, near], [far_depth, near_depth], 1),
                                     ([near, far], [near_depth, far_depth], 0)):
        remaining = iter(maps)
        monkeypatch.setattr(depth_guided, "_infer_depth", lambda _: next(remaining))
        result, labels, confidence = depth_guided.fuse_depth_guided(images, radius=5)
        np.testing.assert_allclose(result[65:120, 85:155], base[65:120, 85:155])
        np.testing.assert_allclose(result[20:25, 20:30], base[20:25, 20:30])
        assert (labels[65:120, 85:155] == near_index).mean() > 0.95
        assert np.isfinite(confidence).all()


def test_depth_guided_uses_several_near_and_far_sources(monkeypatch):
    rng = np.random.default_rng(31)
    base = rng.integers(30, 220, (180, 240, 3)).astype(np.float32) + 0.25
    soft = cv2.GaussianBlur(base, (0, 0), 4)
    near_left, near_right, far_top, far_bottom = [soft.copy() for _ in range(4)]
    near_left[35:155, 55:120] = base[35:155, 55:120]
    near_right[35:155, 120:185] = base[35:155, 120:185]
    far_top[:90] = base[:90]
    far_bottom[90:] = base[90:]
    far_top[35:155, 55:185] = soft[35:155, 55:185]
    far_bottom[35:155, 55:185] = soft[35:155, 55:185]
    near_depth = np.zeros(base.shape[:2], np.float32)
    near_depth[35:155, 55:185] = 2
    far_depth = np.zeros_like(near_depth)
    far_depth[145:] = 2
    remaining = iter([near_depth, near_depth, far_depth, far_depth])
    monkeypatch.setattr(depth_guided, "_infer_depth", lambda _: next(remaining))

    result, labels, _ = depth_guided.fuse_depth_guided(
        [near_left, near_right, far_top, far_bottom], radius=5,
    )
    for area, index in ((np.s_[65:105, 75:100], 0),
                        (np.s_[65:105, 145:165], 1),
                        (np.s_[20:45, 15:35], 2),
                        (np.s_[125:145, 15:35], 3)):
        assert (labels[area] == index).mean() > 0.8
        np.testing.assert_allclose(result[area], base[area], atol=0.01)


def test_landscape_cancellation_leaves_no_output():
    algo = LaplacianPyramid(AlgorithmConfig(stacking_method="landscape"))
    algo.update_image_paths([
        "tests/low_res_images/DSC_0356.jpg",
        "tests/low_res_images/DSC_0358.jpg",
    ])
    algo.stack_images(progress_callback=lambda *_: algo.cancel())
    assert algo.output_image is None


def test_auto_alignment_uses_landscape_for_landscape_methods():
    for method in ("landscape", "landscape_blend", "near_far_cut",
                   "landscape_regions", "landscape_depth"):
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
    diagnostics = algorithm.last_alignment_diagnostics
    assert diagnostics["status"] == "registered"
    assert diagnostics["inliers"] >= 12
    assert 0 < diagnostics["inlier_ratio"] <= 1


def test_landscape_alignment_records_unmatchable_fallback(caplog):
    blank = np.zeros((128, 192, 3), dtype=np.float32)
    algorithm = Algorithm()
    aligned = algorithm.align_image_pair(blank, blank.copy(), alignment_mode="landscape")
    diagnostics = algorithm.last_alignment_diagnostics
    assert diagnostics["status"] == "fallback"
    assert diagnostics["keypoints_reference"] == 0
    assert diagnostics["keypoints_moving"] == 0
    assert diagnostics["reason"] == "No stable focus-overlap features"
    assert diagnostics["fallback_method"] == "identity_low_texture"
    assert "keypoints=0/0" in caplog.text
    assert aligned.shape == blank.shape


def test_landscape_alignment_preserves_framing_for_unrelated_texture():
    rng = np.random.default_rng(903)
    reference = rng.integers(0, 256, (320, 420, 3), dtype=np.uint8).astype(np.float32)
    moving = rng.integers(0, 256, (320, 420, 3), dtype=np.uint8).astype(np.float32)
    algorithm = Algorithm()
    aligned = algorithm.align_image_pair(reference, moving, alignment_mode="landscape")
    assert algorithm.last_alignment_diagnostics["status"] == "fallback"
    assert algorithm.last_alignment_diagnostics["fallback_method"] == "identity_unreliable_match"
    np.testing.assert_array_equal(aligned, moving)
    assert algorithm.last_alignment_mask.all()


def test_roma_correspondences_fit_a_conservative_similarity(monkeypatch):
    rng = np.random.default_rng(11)
    reference = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8).astype(np.float32)
    shifted = cv2.warpAffine(reference, np.float32([[1, 0, 6], [0, 1, -4]]),
                             (240, 180), borderMode=cv2.BORDER_CONSTANT)
    xs, ys = np.meshgrid(np.arange(20, 220, 20), np.arange(20, 160, 20))
    points = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    monkeypatch.setattr(learned_alignment, "correspondences",
                        lambda *_: (points, points + [6, -4], np.ones(len(points))))
    algorithm = Algorithm()
    aligned = algorithm.align_image_pair(reference, shifted, alignment_mode="roma")
    assert algorithm.last_alignment_diagnostics["status"] == "registered"
    assert algorithm.last_alignment_diagnostics["inliers"] >= 12
    assert algorithm.last_alignment_mask.shape == reference.shape[:2]
    assert np.mean(np.abs(aligned[20:-20, 20:-20] - reference[20:-20, 20:-20])) < 1


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


def test_cli_runs_new_region_methods_as_separate_results(tmp_path):
    output = tmp_path / "comparison.tif"
    completed = subprocess.run(
        [sys.executable, "-m", "src.cli", "--input",
         "tests/low_res_images/DSC_0356.jpg", "tests/low_res_images/DSC_0358.jpg",
         "--output", str(output), "--method", "near_far_cut,landscape_regions",
         "--bit-depth", "16"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    for method in ("near_far_cut", "landscape_regions"):
        image = cv2.imread(str(tmp_path / f"comparison_{method}.tif"), cv2.IMREAD_UNCHANGED)
        assert image is not None
        assert image.dtype == np.uint16
