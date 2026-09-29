"""Optional RoMa correspondences for conservative landscape registration.

RoMa estimates matches only; ChimpStackr still fits and applies a single
similarity warp to the original float32 image. The dependency is installed
only in Pixi's ``learned`` or ``learned-cuda`` environment.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


@lru_cache(maxsize=1)
def _model():
    try:
        import torch
        from romatch import roma_outdoor
    except ImportError as error:
        raise RuntimeError(
            "RoMa alignment requires the optional learned environment "
            "(run with 'pixi run -e learned gui' or "
            "'pixi run -e learned-cuda gui')"
        ) from error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def verified_model(**kwargs):
        model = roma_outdoor(device=device, **kwargs)
        checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
        expected = {
            "roma_outdoor.pth": "c7a45c80d41ad788a63c641d1b686d7cb3f297f40097c6f4e75039889e5cc8ba",
            "dinov2_vitl14_pretrain.pth": "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
        }
        for name, digest in expected.items():
            checksum = hashlib.sha256()
            with (checkpoints / name).open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != digest:
                raise RuntimeError(f"RoMa checkpoint hash differs from the pinned {name}")
        return model, device
    if device == "cpu":
        # The official 560/864 setting is prohibitively slow on typical CPUs.
        # GPU inference retains the paper's default resolution.
        return verified_model(coarse_res=280, upsample_res=448)
    return verified_model()


def correspondences(reference, moving):
    """Return corresponding full-image pixel positions and sample certainty."""
    model, device = _model()
    ref = np.clip(reference, 0, 255).astype(np.uint8)
    mov = np.clip(moving, 0, 255).astype(np.uint8)
    with TemporaryDirectory(prefix="chimpstackr-roma-") as directory:
        ref_path = Path(directory) / "reference.png"
        mov_path = Path(directory) / "moving.png"
        if not cv2.imwrite(str(ref_path), ref) or not cv2.imwrite(str(mov_path), mov):
            raise RuntimeError("Could not prepare images for RoMa")
        warp, certainty = model.match(str(ref_path), str(mov_path), device=device)
        matches, certainty = model.sample(warp, certainty)
        pts_ref, pts_mov = model.to_pixel_coordinates(
            matches, reference.shape[0], reference.shape[1],
            moving.shape[0], moving.shape[1],
        )
    pts_ref = pts_ref.detach().cpu().numpy().astype(np.float32)
    pts_mov = pts_mov.detach().cpu().numpy().astype(np.float32)
    certainty = certainty.detach().cpu().numpy().reshape(-1).astype(np.float32)
    finite = (np.isfinite(pts_ref).all(axis=1) & np.isfinite(pts_mov).all(axis=1)
              & np.isfinite(certainty))
    return pts_ref[finite], pts_mov[finite], certainty[finite]
