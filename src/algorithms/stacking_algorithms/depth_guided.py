"""Automatic near/far fusion with a learned relative-depth foreground prior.

Depth Anything V2 Small supplies a coarse geometric prior. A focus comparison
selects which input depicts the near layer, and original aligned source pixels
are composited at full resolution. This is an adaptation for short landscape
stacks, not a reproduction of a published focus-fusion network.
"""

from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import cv2
import numpy as np

from .landscape import focus_score
from .region_fusion import _inputs, fuse_regions


_CHECKPOINT_NAME = "depth_anything_v2_vits.pth"
_CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/"
    "resolve/main/depth_anything_v2_vits.pth"
)
_CHECKPOINT_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
logger = logging.getLogger(__name__)


def _verify_checkpoint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != _CHECKPOINT_SHA256:
        raise RuntimeError(f"Depth Anything V2 Small checkpoint hash differs: {path}")


def _checkpoint_path(torch):
    override = os.environ.get("CHIMPSTACKR_DEPTH_CHECKPOINT")
    path = (Path(override) if override else
            Path(torch.hub.get_dir()) / "checkpoints" / _CHECKPOINT_NAME)
    if not path.exists():
        if override:
            raise RuntimeError(f"Depth checkpoint does not exist: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as output:
                temporary = Path(output.name)
                with urlopen(_CHECKPOINT_URL, timeout=120) as response:
                    while block := response.read(8 * 1024 * 1024):
                        output.write(block)
            _verify_checkpoint(temporary)
            temporary.replace(path)
        except Exception as error:
            raise RuntimeError(
                "Could not download the pinned Depth Anything V2 Small checkpoint; "
                "set CHIMPSTACKR_DEPTH_CHECKPOINT to a verified local copy"
            ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    _verify_checkpoint(path)
    return path


@lru_cache(maxsize=1)
def _model():
    try:
        import torch
        from src.third_party.depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as error:
        raise RuntimeError(
            "Landscape Depth requires PyTorch; run with 'pixi run -e learned gui' "
            "or 'pixi run -e learned-cuda gui'"
        ) from error
    checkpoint = _checkpoint_path(torch)
    model = DepthAnythingV2(encoder="vits", features=64,
                            out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), torch


def _infer_depth(image):
    model, torch = _model()
    source = np.clip(image, 0, 255).astype(np.uint8)
    with torch.inference_mode():
        return model.infer_image(source, input_size=756).astype(np.float32)


def _small_size(shape):
    height, width = shape
    scale = min(1.0, 1536 / max(height, width))
    return max(32, round(width * scale)), max(32, round(height * scale))


def _select_group(images, valid, scores, indices, size):
    """Choose one source per pixel within a focal-layer group."""
    height, width = images[0].shape[:2]
    if len(indices) == 1:
        index = indices[0]
        return images[index], np.full((height, width), index, np.uint8), valid[index]

    smoothed = np.stack([
        cv2.GaussianBlur(scores[index], (0, 0), 4.0) for index in indices
    ])
    small_valid = np.stack([
        cv2.resize(valid[index].astype(np.uint8), size,
                   interpolation=cv2.INTER_NEAREST).astype(bool)
        for index in indices
    ])
    smoothed[~small_valid] = -1.0
    selected = np.asarray(indices, dtype=np.uint8)[np.argmax(smoothed, axis=0)]
    labels = cv2.resize(selected, (width, height), interpolation=cv2.INTER_NEAREST)
    group_valid = np.any(valid[indices], axis=0)
    bad = ~np.take_along_axis(valid, labels[None], axis=0)[0]
    if bad.any():
        for index in indices:
            labels[bad & valid[index]] = index
    result = np.zeros_like(images[0])
    for index in indices:
        chosen = labels == index
        result[chosen] = images[index][chosen]
    return result, labels, group_valid


def fuse_depth_guided(images, valid_masks=None, radius=7):
    """Fuse two to four near/far frames with an inferred foreground layer."""
    images, valid = _inputs(images, valid_masks)
    height, width = images[0].shape[:2]
    size = _small_size((height, width))
    small_images = [cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                    for image in images]
    small_valid = np.stack([
        cv2.resize(mask.astype(np.uint8), size,
                   interpolation=cv2.INTER_NEAREST).astype(bool)
        for mask in valid
    ])
    scores = np.stack([focus_score(image, max(3, radius // 2))
                       for image in small_images])
    scores[~small_valid] = 0.0
    depths = [_infer_depth(image) for image in small_images]

    qualities = []
    foregrounds = []
    global_focus = []
    for index, depth in enumerate(depths):
        source_valid = small_valid[index]
        if not source_valid.any():
            qualities.append(-np.inf)
            foregrounds.append(np.zeros_like(source_valid))
            global_focus.append(0.0)
            continue
        threshold = np.percentile(depth[source_valid], 85)
        foreground = (depth >= threshold) & source_valid
        foregrounds.append(foreground)
        global_focus.append(float(scores[index][source_valid].mean()))
        background = (~foreground) & source_valid
        if foreground.sum() < 20 or background.sum() < 20:
            qualities.append(-np.inf)
            continue
        rival = np.max(np.delete(scores, index, axis=0), axis=0)
        margin = (scores[index] - rival) / (scores[index] + rival + 1e-5)
        qualities.append(float(margin[foreground].mean() - margin[background].mean()))
    # A far-focused field can have a sharp strip of near grass. Its depth map
    # may call that strip foreground, even though it is the wrong near layer.
    # Penalize globally sharp sources when identifying the foreground anchor.
    positive_focus = [value for value in global_focus if value > 0]
    smallest_focus = max(min(positive_focus), 1e-6) if positive_focus else 1e-6
    adjusted = [quality - np.log(max(mean, 1e-6) / smallest_focus)
                for quality, mean in zip(qualities, global_focus)]
    near_anchor = int(np.argmax(adjusted))
    logger.debug("Landscape Depth near-anchor qualities: %s adjusted: %s",
                 qualities, adjusted)
    if not np.isfinite(qualities[near_anchor]):
        raise ValueError("Depth-guided fusion found no valid foreground source")
    ranked = sorted(adjusted, reverse=True)
    if ranked[0] < 0.1 or (len(images) == 2 and ranked[0] - ranked[1] < 0.08):
        logger.info("Landscape Depth used region fallback: ambiguous near/far focus evidence")
        return fuse_regions(images, valid, radius)

    anchor_foreground = foregrounds[near_anchor]
    near_indices = []
    for index, foreground in enumerate(foregrounds):
        overlap = np.count_nonzero(anchor_foreground & foreground)
        union = np.count_nonzero(anchor_foreground | foreground)
        if (index == near_anchor or
                (union and overlap / union > 0.35 and adjusted[index] > 0.1)):
            near_indices.append(index)
    far_indices = [index for index in range(len(images)) if index not in near_indices]
    if not far_indices:
        logger.info("Landscape Depth used region fallback: no distinct far source")
        return fuse_regions(images, valid, radius)
    logger.debug("Landscape Depth source groups: near=%s far=%s", near_indices, far_indices)

    near_image, near_labels, near_valid = _select_group(
        images, valid, scores, near_indices, size,
    )
    far_image, far_labels, far_valid = _select_group(
        images, valid, scores, far_indices, size,
    )

    foregrounds_75 = []
    for index in near_indices:
        depth = depths[index]
        threshold = np.percentile(depth[small_valid[index]], 75)
        foregrounds_75.append(depth >= threshold)
    small_foreground = np.mean(foregrounds_75, axis=0) >= 0.5
    foreground = cv2.resize(small_foreground.astype(np.float32),
                            (width, height), interpolation=cv2.INTER_NEAREST)
    alpha = cv2.GaussianBlur(foreground, (0, 0), 2.0)
    alpha[~near_valid & far_valid] = 0.0
    alpha[near_valid & ~far_valid] = 1.0
    if not np.all(near_valid | far_valid):
        raise ValueError("Depth-guided fusion has no valid source at some pixels")
    result = (near_image * alpha[:, :, None]
              + far_image * (1.0 - alpha[:, :, None])).astype(np.float32)
    labels = np.where(alpha >= 0.5, near_labels, far_labels).astype(np.uint8)
    confidence = np.clip(1.0 - 2.0 * np.abs(foreground - alpha),
                         0, 1).astype(np.float32)
    return result, labels, confidence

