"""Focus-map fusion for short, mostly static landscape stacks.

The decision map is estimated at multiple scales, then all output pixels are
composited from the original float32 inputs. This module has no Qt dependency.
"""

import cv2
import numpy as np


def _gray(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Landscape fusion expects grayscale or three-channel BGR images")


def focus_score(image, radius=7):
    """Measure locally coherent detail without converting the source to 8-bit."""
    gray = _gray(image)
    radius = max(1, int(radius))
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = np.abs(cv2.Laplacian(smooth, cv2.CV_32F, ksize=3))
    k = radius * 2 + 1
    local = cv2.boxFilter(lap, cv2.CV_32F, (k, k))
    broad = cv2.boxFilter(lap, cv2.CV_32F, (k * 3, k * 3))
    luminance = cv2.boxFilter(gray, cv2.CV_32F, (k, k))
    score = (0.75 * local + 0.25 * broad) / (luminance + 24.0)
    return score.astype(np.float32)


def _guided_filter(guide, signal, radius, epsilon=0.0025):
    """Fast grayscale guided filtering with a scene-edge guide in [0, 1]."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean_g = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(signal, -1, k)
    covariance = cv2.boxFilter(guide * signal, -1, k) - mean_g * mean_p
    variance = cv2.boxFilter(guide * guide, -1, k) - mean_g * mean_g
    a = covariance / (variance + epsilon)
    b = mean_p - a * mean_g
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def fuse(images, valid_masks=None, radius=7, blend=False):
    """Fuse 2-4 registered images. Return (float32 image, index map, confidence).

    `valid_masks` mark pixels supplied by each original source, excluding
    black warp borders. Index zero wins exact ties in featureless regions.
    """
    images = [np.asarray(img, dtype=np.float32) for img in images]
    if not 2 <= len(images) <= 4:
        raise ValueError("Landscape fusion requires two to four frames")
    shape = images[0].shape
    if any(img.shape != shape for img in images):
        raise ValueError("Landscape fusion requires images with equal dimensions")

    h, w = shape[:2]
    if valid_masks is None:
        valid_masks = [np.ones((h, w), dtype=bool) for _ in images]
    if len(valid_masks) != len(images):
        raise ValueError("One validity mask is required per frame")
    valid = np.stack([np.asarray(m, dtype=bool) for m in valid_masks])
    if valid.shape != (len(images), h, w):
        raise ValueError("Validity masks must match image height and width")

    scores = np.stack([focus_score(img, radius) for img in images])
    scores[~valid] = 0.0
    # A focus comparison should only consider genuine source pixels.
    available = valid.any(axis=0)
    if not available.all():
        raise ValueError("No valid source frame at some output pixels")

    # A wide support makes the large foreground/background regions coherent;
    # fine texture still enters through the original local score.
    sigma = max(2.0, min(h, w) / 180.0)
    broad = np.stack([
        cv2.GaussianBlur(score, (0, 0), sigmaX=sigma) for score in scores
    ])
    evidence = 0.65 * broad + 0.35 * scores
    evidence[~valid] = 0.0
    provisional = np.argmax(np.where(valid, evidence, -1.0), axis=0)

    # Construct the guide from the provisional sharpest source, so an
    # out-of-focus reference does not define the final region boundaries.
    guide = np.zeros((h, w), np.float32)
    for i, img in enumerate(images):
        selected = provisional == i
        if selected.any():
            gray = _gray(img)
            guide[selected] = gray[selected] / 255.0
    guide = np.clip(guide, 0.0, 1.0)

    denominator = evidence.sum(axis=0) + 1e-8
    guided = np.stack([
        _guided_filter(guide, evidence[i] / denominator, max(2, radius * 2))
        for i in range(len(images))
    ])
    guided = np.maximum(guided, 0.0)
    guided[~valid] = 0.0
    labels = np.argmax(np.where(valid, guided, -1.0), axis=0).astype(np.uint8)

    ranked = np.sort(evidence, axis=0)
    confidence = (ranked[-1] - ranked[-2]) / (ranked[-1] + ranked[-2] + 1e-8)
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)

    result = np.zeros_like(images[0], dtype=np.float32)
    if blend:
        # Blend a narrow seam only if the competing sources have similar
        # focus evidence and little photometric disagreement. This gives a
        # separate natural-transition option without ghosting broad sky or
        # water regions.
        weights = np.stack([
            cv2.GaussianBlur((labels == i).astype(np.float32), (5, 5), 0)
            for i in range(len(images))
        ])
        weights *= valid
        weights /= weights.sum(axis=0, keepdims=True) + 1e-8
        primary_gray = guide * 255.0
        for i, img in enumerate(images):
            similar_focus = evidence[i] >= 0.6 * ranked[-1]
            similar_pixels = np.abs(_gray(img) - primary_gray) < 12.0
            weights[i] *= similar_focus & similar_pixels
        weights /= weights.sum(axis=0, keepdims=True) + 1e-8
        # The mask can become empty after gating; retain the source winner.
        empty = weights.sum(axis=0) < 0.5
        for i in range(len(images)):
            weights[i][empty] = labels[empty] == i
            if images[0].ndim == 3:
                result += images[i] * weights[i, :, :, None]
            else:
                result += images[i] * weights[i]
    else:
        for i, img in enumerate(images):
            chosen = labels == i
            result[chosen] = img[chosen]
    return result, labels, confidence
