"""Region-consistent landscape fusion using graph-cut focus labels.

The binary method uses an edge-weighted cut; the multi-frame method uses
alpha expansion. Both retain pixels from the original aligned float32 sources.
They are inspired by graph-cut focus segmentation research, not reproductions
of a particular trained model.
"""

import cv2
import maxflow
import numpy as np

from .landscape import _gray, focus_score


def _inputs(images, valid_masks):
    images = [np.asarray(image, dtype=np.float32) for image in images]
    if not 2 <= len(images) <= 4:
        raise ValueError("Region fusion requires two to four images")
    if any(image.shape != images[0].shape for image in images):
        raise ValueError("Region fusion requires equal image shapes")
    h, w = images[0].shape[:2]
    if valid_masks is None:
        valid_masks = [np.ones((h, w), dtype=bool) for _ in images]
    if len(valid_masks) != len(images):
        raise ValueError("One validity mask is required per image")
    valid = np.stack([np.asarray(mask, dtype=bool) for mask in valid_masks])
    if valid.shape != (len(images), h, w) or not valid.any(axis=0).all():
        raise ValueError("Every output pixel needs at least one valid source")
    return images, valid


def _scores(images, valid, radius):
    scores = np.stack([focus_score(image, radius) for image in images])
    scores[~valid] = 0.0
    return scores


def _downsample_shape(shape, maximum):
    h, w = shape
    scale = min(1.0, maximum / max(h, w))
    return max(1, round(w * scale)), max(1, round(h * scale))


def _unary(scores, valid, size):
    small_scores = np.stack([
        cv2.resize(score, size, interpolation=cv2.INTER_AREA)
        for score in scores
    ])
    small_valid = np.stack([
        cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        for mask in valid
    ])
    # Compare relative focus, avoiding exposure-dependent absolute thresholds.
    scale = np.maximum(small_scores.sum(axis=0), 1e-5)
    probabilities = (small_scores + 0.005 * scale) / (scale * (1 + 0.005 * len(scores)))
    unary = -np.log(np.maximum(probabilities, 1e-6)).astype(np.float64)
    unary[~small_valid] = 1000.0
    return unary


def _assemble(images, valid, scores, labels):
    h, w = labels.shape
    # Full-resolution validity always wins over low-resolution optimization.
    invalid = ~np.take_along_axis(valid, labels[None], axis=0)[0]
    if invalid.any():
        candidates = np.where(valid, scores, -1.0)
        labels[invalid] = np.argmax(candidates[:, invalid], axis=0)

    ordered = np.sort(np.where(valid, scores, 0.0), axis=0)
    confidence = (ordered[-1] - ordered[-2]) / (ordered[-1] + ordered[-2] + 1e-6)
    result = np.empty_like(images[0], dtype=np.float32)
    for index, image in enumerate(images):
        chosen = labels == index
        result[chosen] = image[chosen]
    return result, labels.astype(np.uint8), np.clip(confidence, 0, 1).astype(np.float32)


def fuse_near_far(images, valid_masks=None, radius=7):
    """Edge-aware binary cut for a near-focused and far-focused pair."""
    images, valid = _inputs(images, valid_masks)
    if len(images) != 2:
        raise ValueError("Near/Far Cut requires exactly two images")
    h, w = images[0].shape[:2]
    scores = _scores(images, valid, radius)
    size = _downsample_shape((h, w), 1536)
    unary = _unary(scores, valid, size)

    # Either source may contain a reliable object boundary. Using their
    # strongest gradient lowers the seam penalty at that boundary.
    small_gray = [cv2.resize(_gray(image), size, interpolation=cv2.INTER_AREA)
                  for image in images]
    gradients = [cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                               cv2.Sobel(gray, cv2.CV_32F, 0, 1))
                 for gray in small_gray]
    edge = np.maximum(gradients[0], gradients[1])
    edge_scale = max(float(np.percentile(edge, 90)), 1.0)
    strength = (0.04 + 0.35 * np.exp(-edge / edge_scale)).astype(np.float64)
    horizontal = np.zeros_like(strength)
    vertical = np.zeros_like(strength)
    horizontal[:, :-1] = 0.5 * (strength[:, :-1] + strength[:, 1:])
    vertical[:-1] = 0.5 * (strength[:-1] + strength[1:])

    graph = maxflow.Graph[float]()
    nodes = graph.add_grid_nodes((size[1], size[0]))
    # Source label 0 costs unary[0]; sink label 1 costs unary[1].
    graph.add_grid_tedges(nodes, unary[1], unary[0])
    right = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    down = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    graph.add_grid_edges(nodes, horizontal, structure=right, symmetric=True)
    graph.add_grid_edges(nodes, vertical, structure=down, symmetric=True)
    graph.maxflow()
    small_labels = graph.get_grid_segments(nodes).astype(np.uint8)
    labels = cv2.resize(small_labels, (w, h), interpolation=cv2.INTER_NEAREST)

    # Restore strongly evidenced thin details lost by the coarse graph.
    ratio = (scores[0] + 1e-4) / (scores[1] + 1e-4)
    labels[(ratio > 3.0) & valid[0]] = 0
    labels[(ratio < 1 / 3.0) & valid[1]] = 1
    return _assemble(images, valid, scores, labels)


def fuse_regions(images, valid_masks=None, radius=7):
    """Multi-label alpha expansion for two to four landscape frames."""
    images, valid = _inputs(images, valid_masks)
    h, w = images[0].shape[:2]
    scores = _scores(images, valid, radius)
    size = _downsample_shape((h, w), 1024)
    unary = np.moveaxis(_unary(scores, valid, size), 0, -1)
    count = len(images)
    pairwise = np.full((count, count), 0.3, dtype=np.float64)
    np.fill_diagonal(pairwise, 0.0)
    small_labels = maxflow.fastmin.aexpansion_grid(unary, pairwise, max_cycles=4)
    labels = cv2.resize(small_labels.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST)

    # Source pixels with an unambiguous focus margin may define small
    # structures that are sub-pixel in the optimization image.
    ranked = np.sort(scores, axis=0)
    confident = ranked[-1] > 3.0 * (ranked[-2] + 1e-4)
    local = np.argmax(np.where(valid, scores, -1.0), axis=0)
    labels[confident] = local[confident]
    return _assemble(images, valid, scores, labels)
