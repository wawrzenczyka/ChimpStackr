"""
    Main pyramid stacking + image alignment algorithm(s).
    Supports translation-only and full rotation+scale+translation alignment.
    Pipeline operates in float32 to preserve full bit depth.
"""
import math
import threading
import os
import cv2
import numpy as np

# Enable OpenCV multi-threading
_cv_threads = os.cpu_count() or 4
cv2.setNumThreads(_cv_threads)

try:
    import numba.cuda as cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

import src.algorithms.dft_imreg as dft_imreg
import src.ImageLoadingHandler as ImageLoadingHandler
import src.algorithms.stacking_algorithms.cpu as CPU
import src.algorithms.learned_alignment as learned_alignment

try:
    import src.algorithms.stacking_algorithms.gpu as GPU
    HAS_GPU = True
except Exception:
    HAS_GPU = False


class Algorithm:
    def __init__(self):
        self.ImageLoadingHandler = ImageLoadingHandler.ImageLoadingHandler()
        self.DFT_Imreg = dft_imreg.im_reg()
        self.DFT_Imreg.last_shift = (0.0, 0.0)
        self.useGpu = False
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused (set = running)
        self.alignment_shifts = []  # Track all (x, y) shifts for auto-crop
        self._ref_gray_cache = (None, None)  # (id(ref_im), gray) cache
        self.last_alignment_mask = None
        self.last_alignment_diagnostics = {}

    def cancel(self):
        self._cancel_event.set()
        self._pause_event.set()

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def reset_cancel(self):
        self._cancel_event.clear()
        self._pause_event.set()

    @property
    def is_cancelled(self):
        return self._cancel_event.is_set()

    @property
    def is_paused(self):
        return not self._pause_event.is_set()

    def wait_if_paused(self):
        """Block until unpaused. Times out after 60s to prevent deadlocks."""
        while not self._pause_event.wait(timeout=60.0):
            if self._cancel_event.is_set():
                return

    def toggle_cpu_gpu(self, use_gpu, selected_gpu_id):
        if use_gpu and HAS_CUDA and HAS_GPU:
            cuda.select_device(selected_gpu_id)
            self.useGpu = True
        else:
            self.useGpu = False

    def load_image(self, path):
        """Load image from path as float32 (preserving full bit depth)."""
        return self.ImageLoadingHandler.read_image_as_float32(path)

    @staticmethod
    def _match_dimensions(ref_im, im_to_align):
        """Resize im_to_align to match ref_im dimensions if they differ.

        Handles mixed image dimensions (#29) by center-cropping or padding
        the image to align to match the reference dimensions. Uses
        center-crop for larger images and zero-padding for smaller ones.
        """
        if ref_im.shape[:2] == im_to_align.shape[:2]:
            return im_to_align

        rh, rw = ref_im.shape[:2]
        ah, aw = im_to_align.shape[:2]

        # If dimensions are very different (>2x), resize to match
        if ah > rh * 2 or aw > rw * 2 or ah < rh // 2 or aw < rw // 2:
            return cv2.resize(im_to_align, (rw, rh), interpolation=cv2.INTER_AREA)

        # Center-crop if larger, zero-pad if smaller
        result = np.zeros_like(ref_im)
        # Source region (from im_to_align)
        sy = max(0, (ah - rh) // 2)
        sx = max(0, (aw - rw) // 2)
        # Destination region (in result)
        dy = max(0, (rh - ah) // 2)
        dx = max(0, (rw - aw) // 2)
        # Copy region size
        ch = min(rh - dy, ah - sy)
        cw = min(rw - dx, aw - sx)
        result[dy:dy+ch, dx:dx+cw] = im_to_align[sy:sy+ch, sx:sx+cw]
        return result

    def align_image_pair(self, ref_im, im_to_align, scale_factor=10,
                         coarse_fine=False, use_rst=False, alignment_mode="translation"):
        """
        Align im_to_align to ref_im.
        Returns float32 aligned image.

        alignment_mode: "translation", "euclidean", "similarity", "affine", "landscape", "roma"
        use_rst: legacy flag — if True and alignment_mode not explicitly set, uses euclidean
        """
        self.last_alignment_mask = None
        self.last_alignment_diagnostics = {}
        # Handle path loading
        if isinstance(ref_im, str) and isinstance(im_to_align, str) and ref_im == im_to_align:
            image = self.load_image(im_to_align)
            self.last_alignment_mask = np.ones(image.shape[:2], dtype=bool)
            return image
        if isinstance(ref_im, str):
            ref_im = self.load_image(ref_im)
        if isinstance(im_to_align, str):
            im_to_align = self.load_image(im_to_align)

        # Handle different image dimensions (#29)
        if ref_im is not None and im_to_align is not None:
            im_to_align = self._match_dimensions(ref_im, im_to_align)

        # Legacy compat: use_rst maps to euclidean if no explicit mode
        if use_rst and alignment_mode == "translation":
            alignment_mode = "euclidean"

        if alignment_mode == "landscape":
            result = self._align_landscape(ref_im, im_to_align, scale_factor)
        elif alignment_mode == "roma":
            result = self._align_roma(ref_im, im_to_align, scale_factor)
        elif alignment_mode == "similarity":
            result = self._align_similarity(ref_im, im_to_align, scale_factor)
        elif alignment_mode == "affine":
            result = self._align_affine(ref_im, im_to_align, scale_factor)
        elif alignment_mode == "euclidean":
            result = self._align_rst(ref_im, im_to_align, scale_factor)
        else:
            result = self._align_translation(ref_im, im_to_align, scale_factor, coarse_fine)

        return result

    def _align_roma(self, ref_im, im_to_align, scale_factor):
        """Fit a guarded global similarity from RoMa's dense correspondences."""
        diagnostics = {"mode": "roma", "status": "matching"}
        self.last_alignment_diagnostics = diagnostics
        try:
            points_ref, points_mov, certainty = learned_alignment.correspondences(
                ref_im, im_to_align,
            )
            diagnostics["matches"] = len(points_ref)
            if len(points_ref) < 12:
                raise ValueError("RoMa produced too few matches")
            # Preserve the best high-confidence matches while leaving enough
            # spatial candidates for the robust fit.
            threshold = np.quantile(certainty, 0.35)
            selected = certainty >= threshold
            points_ref = points_ref[selected]
            points_mov = points_mov[selected]
            diagnostics["selected_matches"] = len(points_ref)
            # Correspondences are returned in original-image pixels. A fixed
            # three-pixel threshold is too strict for a 6000-pixel landscape.
            reprojection_threshold = max(3.0, 2.5 * max(ref_im.shape[:2]) / 2048.0)
            warp, inliers = cv2.estimateAffinePartial2D(
                points_ref, points_mov, method=cv2.RANSAC,
                ransacReprojThreshold=reprojection_threshold,
                maxIters=5000, confidence=0.995,
            )
            diagnostics["reprojection_threshold"] = reprojection_threshold
            count = int(inliers.sum()) if inliers is not None else 0
            diagnostics["inliers"] = count
            diagnostics["inlier_ratio"] = count / len(points_ref)
            if warp is None or count < 12 or diagnostics["inlier_ratio"] < 0.25:
                raise ValueError("RoMa similarity fit has too few reliable inliers")
            scale = float(np.hypot(warp[0, 0], warp[1, 0]))
            diagnostics["scale"] = scale
            if not 0.75 < scale < 1.33:
                raise ValueError("RoMa similarity scale is implausible")
            h, w = ref_im.shape[:2]
            result = cv2.warpAffine(
                im_to_align, warp, (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            self.last_alignment_mask = self._valid_mask_for_warp(warp, im_to_align.shape)
            self._track_warp_shifts(warp, im_to_align.shape)
            diagnostics["status"] = "registered"
            return result
        except (cv2.error, ValueError) as error:
            import logging
            logging.getLogger(__name__).warning("RoMa alignment fallback: %s", error)
            result = self._align_landscape(ref_im, im_to_align, scale_factor)
            self.last_alignment_diagnostics["requested_mode"] = "roma"
            self.last_alignment_diagnostics["roma_reason"] = str(error)
            return result

    def _align_landscape(self, ref_im, im_to_align, scale_factor):
        """Focus-aware SIFT similarity alignment with guarded ECC refinement.

        The model estimates correspondences at a bounded resolution, then
        applies one full-resolution warp to the original float32 source.
        Learned matchers can later replace SIFT without changing the warp and
        validity-mask contract.
        """
        ref = self._to_gray_u8(ref_im)
        moving = self._to_gray_u8(im_to_align)
        h, w = ref.shape
        scale = min(1.0, 2048.0 / max(h, w))
        if scale < 1.0:
            size = (max(16, round(w * scale)), max(16, round(h * scale)))
            ref_small = cv2.resize(ref, size, interpolation=cv2.INTER_AREA)
            mov_small = cv2.resize(moving, size, interpolation=cv2.INTER_AREA)
            sx, sy = size[0] / w, size[1] / h
        else:
            ref_small, mov_small = ref, moving
            sx = sy = 1.0

        diagnostics = {"mode": "landscape", "status": "matching"}
        self.last_alignment_diagnostics = diagnostics
        try:
            sift = cv2.SIFT_create(nfeatures=5000)
            kp_ref, des_ref = sift.detectAndCompute(ref_small, None)
            kp_mov, des_mov = sift.detectAndCompute(mov_small, None)
            diagnostics["keypoints_reference"] = len(kp_ref or ())
            diagnostics["keypoints_moving"] = len(kp_mov or ())
            if des_ref is None or des_mov is None:
                raise ValueError("No stable focus-overlap features")
            matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_ref, des_mov, k=2)
            good = [m for pair in matches if len(pair) == 2
                    for m, n in [pair] if m.distance < 0.7 * n.distance]
            diagnostics["ratio_matches"] = len(good)
            if len(good) < 12:
                raise ValueError("Too few shared features")

            points_ref = np.float32([kp_ref[m.queryIdx].pt for m in good])
            points_mov = np.float32([kp_mov[m.trainIdx].pt for m in good])
            coarse, inliers = cv2.estimateAffinePartial2D(
                points_ref, points_mov, method=cv2.RANSAC,
                ransacReprojThreshold=2.5, maxIters=4000, confidence=0.995,
            )
            inlier_count = int(inliers.sum()) if inliers is not None else 0
            diagnostics["inliers"] = inlier_count
            diagnostics["inlier_ratio"] = inlier_count / len(good)
            if coarse is None or inlier_count < 12:
                raise ValueError("Similarity fit has too few inliers")
            ratio = diagnostics["inlier_ratio"]
            fit_scale = float(np.hypot(coarse[0, 0], coarse[1, 0]))
            diagnostics["scale"] = fit_scale
            if ratio < 0.3 or not 0.75 < fit_scale < 1.33:
                raise ValueError("Similarity fit is implausible")

            # `coarse` maps reference pixels to moving pixels because all
            # final warps use WARP_INVERSE_MAP.
            full = coarse.astype(np.float64)
            full[0, 1] *= sy / sx
            full[1, 0] *= sx / sy
            full[0, 2] /= sx
            full[1, 2] /= sy
            full = full.astype(np.float32)

            # Refine only shared textured regions. ECC over an entire focus
            # pair may be pulled toward the differently blurred background.
            pre = cv2.warpAffine(
                mov_small, coarse, ref_small.shape[::-1],
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            lap_ref = cv2.boxFilter(np.abs(cv2.Laplacian(ref_small, cv2.CV_32F)),
                                    -1, (11, 11))
            lap_pre = cv2.boxFilter(np.abs(cv2.Laplacian(pre, cv2.CV_32F)),
                                    -1, (11, 11))
            mask = ((lap_ref > np.percentile(lap_ref, 45)) &
                    (lap_pre > np.percentile(lap_pre, 45))).astype(np.uint8) * 255
            if cv2.countNonZero(mask) > 2000:
                residual = np.eye(2, 3, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
                try:
                    _, residual = cv2.findTransformECC(
                        ref_small, pre, residual, cv2.MOTION_EUCLIDEAN,
                        criteria, inputMask=mask, gaussFiltSize=5,
                    )
                    refined = cv2.warpAffine(
                        pre, residual, ref_small.shape[::-1],
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    before = float(cv2.mean(cv2.absdiff(ref_small, pre), mask)[0])
                    after = float(cv2.mean(cv2.absdiff(ref_small, refined), mask)[0])
                    if after < before:
                        # First map reference -> prealigned, then prealigned
                        # -> moving. Transform composition is C @ R.
                        C = np.vstack([coarse, [0, 0, 1]])
                        R = np.vstack([residual, [0, 0, 1]])
                        combined = (C @ R)[:2]
                        full = combined.astype(np.float64)
                        full[0, 1] *= sy / sx
                        full[1, 0] *= sx / sy
                        full[0, 2] /= sx
                        full[1, 2] /= sy
                        full = full.astype(np.float32)
                except cv2.error:
                    pass

            result = cv2.warpAffine(
                im_to_align, full, (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            self.last_alignment_mask = cv2.warpAffine(
                np.ones(im_to_align.shape[:2], dtype=np.uint8), full, (w, h),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(bool)
            self._track_warp_shifts(full, im_to_align.shape)
            diagnostics["status"] = "registered"
            return result
        except (cv2.error, ValueError, AttributeError) as error:
            # Fall back to the established similarity path, which also
            # records a valid-pixel mask for its chosen warp.
            import logging
            diagnostics["status"] = "fallback"
            diagnostics["reason"] = str(error)
            logging.getLogger(__name__).warning(
                "Landscape alignment fallback: %s; keypoints=%s/%s, "
                "ratio_matches=%s, inliers=%s, inlier_ratio=%s",
                error, diagnostics.get("keypoints_reference", "?"),
                diagnostics.get("keypoints_moving", "?"),
                diagnostics.get("ratio_matches", "?"),
                diagnostics.get("inliers", "?"),
                diagnostics.get("inlier_ratio", "?"),
            )
            if np.std(ref_small) < 1.0 or np.std(mov_small) < 1.0:
                # Registration has no reliable signal in a nearly uniform
                # source. The DFT fallback can produce NaNs here.
                diagnostics["fallback_method"] = "identity_low_texture"
                self.last_alignment_mask = np.ones(ref_im.shape[:2], dtype=bool)
                self._track_warp_shifts(np.eye(2, 3, dtype=np.float32), im_to_align.shape)
                return im_to_align.copy()
            diagnostics["fallback_method"] = "similarity"
            result = self._align_similarity(ref_im, im_to_align, scale_factor)
            return result

    def _get_ref_gray(self, ref_im):
        """Get cached grayscale of reference image (avoids redundant cvtColor)."""
        ref_id = id(ref_im)
        if self._ref_gray_cache[0] != ref_id:
            if ref_im.ndim == 3:
                gray = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
            else:
                gray = ref_im
            self._ref_gray_cache = (ref_id, gray)
        return self._ref_gray_cache[1]

    def _align_translation(self, ref_im, im_to_align, scale_factor, coarse_fine):
        """Translation-only alignment using DFT phase correlation."""
        ref_gray = self._get_ref_gray(ref_im)

        if coarse_fine and min(ref_im.shape[:2]) > 1000:
            small_ref = cv2.resize(ref_im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            small_align = cv2.resize(im_to_align, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            self.DFT_Imreg.register_image_translation(
                small_ref, small_align, scale_factor=max(1, scale_factor // 2)
            )
            del small_ref, small_align

        result = self.DFT_Imreg.register_image_translation(
            ref_im, im_to_align, scale_factor=scale_factor,
            ref_gray=ref_gray,
        )
        self.alignment_shifts.append(self.DFT_Imreg.last_shift)
        x_shift, y_shift = self.DFT_Imreg.last_shift
        h, w = im_to_align.shape[:2]
        self.last_alignment_mask = cv2.warpAffine(
            np.ones((h, w), np.uint8),
            np.float32([[1, 0, x_shift], [0, 1, y_shift]]), (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        return result

    def _align_rst(self, ref_im, im_to_align, scale_factor):
        """
        Rotation + Scale + Translation alignment using multi-scale ECC.
        Uses MOTION_EUCLIDEAN (translation + rotation, 3 DOF) which is
        the right model for focus stacking — no shear, no anisotropic scale.
        Focus breathing (uniform scale) is handled by ORB feature matching fallback.
        """
        # Convert to grayscale (reuse cached ref grayscale when available)
        ref_gray = self._get_ref_gray(ref_im)
        if im_to_align.ndim == 3:
            align_gray = cv2.cvtColor(im_to_align, cv2.COLOR_BGR2GRAY)
        else:
            align_gray = im_to_align

        # Ensure uint8
        if ref_gray.dtype != np.uint8:
            ref_u8 = np.clip(ref_gray, 0, 255).astype(np.uint8)
            align_u8 = np.clip(align_gray, 0, 255).astype(np.uint8)
        else:
            ref_u8 = ref_gray
            align_u8 = align_gray

        h_full, w_full = ref_u8.shape[:2]

        # Multi-scale ECC: coarse → fine
        # Level 0: 1/4 res (coarse alignment)
        # Level 1: 1/2 res (refine)
        # Level 2: full or capped at 2048px (final refinement)
        scales = []
        for max_dim in [512, 1024, min(2048, max(h_full, w_full))]:
            s = min(max_dim / max(h_full, w_full), 1.0)
            if not scales or s > scales[-1]:
                scales.append(s)

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            for i, s in enumerate(scales):
                if s < 1.0:
                    ref_s = cv2.resize(ref_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    align_s = cv2.resize(align_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                else:
                    ref_s = ref_u8
                    align_s = align_u8

                # Scale warp matrix translation from previous level
                if i > 0:
                    prev_s = scales[i - 1]
                    warp_matrix[0, 2] *= (s / prev_s)
                    warp_matrix[1, 2] *= (s / prev_s)

                # More iterations at coarse, fewer at fine
                max_iter = 200 if i == 0 else 100
                gauss = 11 if i == 0 else 5
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    max_iter, 1e-6,
                )

                # EUCLIDEAN: translation + rotation (3 DOF) — correct for focus stacking
                _, warp_matrix = cv2.findTransformECC(
                    ref_s, align_s, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria,
                    inputMask=None, gaussFiltSize=gauss,
                )

            # Scale translation back to full resolution
            final_s = scales[-1]
            if final_s < 1.0:
                warp_matrix[0, 2] /= final_s
                warp_matrix[1, 2] /= final_s

        except cv2.error:
            # ECC failed — fall back to translation-only DFT
            return self._align_translation(ref_im, im_to_align, scale_factor, False)

        # Apply the warp to the full-resolution color image
        h, w = im_to_align.shape[:2]
        result = cv2.warpAffine(
            im_to_align, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        self.last_alignment_mask = self._valid_mask_for_warp(warp_matrix, im_to_align.shape)
        self._track_warp_shifts(warp_matrix, im_to_align.shape)
        return result

    def _track_warp_shifts(self, warp_matrix, shape):
        """Track max corner displacement for auto-crop (shared by RST/similarity/affine)."""
        h, w = shape[:2]
        M = warp_matrix
        max_dx = max_dy = 0.0
        for cx, cy in [(0, 0), (w, 0), (0, h), (w, h)]:
            tx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
            ty = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
            max_dx = max(max_dx, abs(tx))
            max_dy = max(max_dy, abs(ty))
        self.alignment_shifts.append((max_dx, max_dy))

    @staticmethod
    def _valid_mask_for_warp(warp_matrix, shape):
        h, w = shape[:2]
        return cv2.warpAffine(
            np.ones((h, w), dtype=np.uint8), warp_matrix, (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)

    def _to_gray_u8(self, im):
        """Convert image to grayscale uint8 for feature matching / ECC."""
        if im.ndim == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    def _feature_match(self, ref_u8, align_u8, mode="similarity"):
        """
        ORB feature matching → estimateAffinePartial2D (4 DOF) or estimateAffine2D (6 DOF).
        Returns 2x3 warp matrix or None if insufficient matches.
        """
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(ref_u8, None)
        kp2, des2 = orb.detectAndCompute(align_u8, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        min_matches = 6 if mode == "similarity" else 10
        if len(good) < min_matches:
            return None

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_align = np.float32([kp2[m.trainIdx].pt for m in good])

        # estimateAffine*2D(src, dst) computes transform FROM src TO dst.
        # warpAffine with WARP_INVERSE_MAP expects the map FROM dst TO src.
        # We want to warp align→ref space, so the inverse map is ref→align.
        # Therefore: estimate(ref→align) = estimate(pts_ref, pts_align)
        if mode == "affine":
            M, inliers = cv2.estimateAffine2D(
                pts_ref, pts_align,
                method=cv2.RANSAC, ransacReprojThreshold=3.0,
                maxIters=2000, confidence=0.99,
            )
        else:
            M, inliers = cv2.estimateAffinePartial2D(
                pts_ref, pts_align,
                method=cv2.RANSAC, ransacReprojThreshold=3.0,
                maxIters=2000, confidence=0.99,
            )

        if M is None or (inliers is not None and inliers.sum() < min_matches):
            return None
        return M.astype(np.float32)

    def _ecc_refine(self, ref_u8, prealigned_u8, motion_type=cv2.MOTION_EUCLIDEAN):
        """
        Multi-scale ECC refinement on a pre-aligned image.
        Returns residual 2x3 warp matrix (identity if ECC fails).
        """
        h_full, w_full = ref_u8.shape[:2]
        scales = []
        for max_dim in [512, 1024, min(2048, max(h_full, w_full))]:
            s = min(max_dim / max(h_full, w_full), 1.0)
            if not scales or s > scales[-1]:
                scales.append(s)

        warp = np.eye(2, 3, dtype=np.float32)

        try:
            for i, s in enumerate(scales):
                if s < 1.0:
                    ref_s = cv2.resize(ref_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    align_s = cv2.resize(prealigned_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                else:
                    ref_s = ref_u8
                    align_s = prealigned_u8

                if i > 0:
                    prev_s = scales[i - 1]
                    warp[0, 2] *= (s / prev_s)
                    warp[1, 2] *= (s / prev_s)

                max_iter = 200 if i == 0 else 100
                gauss = 11 if i == 0 else 5
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1e-6)

                _, warp = cv2.findTransformECC(
                    ref_s, align_s, warp, motion_type, criteria,
                    inputMask=None, gaussFiltSize=gauss,
                )

            final_s = scales[-1]
            if final_s < 1.0:
                warp[0, 2] /= final_s
                warp[1, 2] /= final_s

        except cv2.error:
            pass  # Return identity (no refinement)

        return warp

    def _align_similarity(self, ref_im, im_to_align, scale_factor):
        """
        4 DOF similarity alignment: tx, ty, rotation, uniform scale.
        Hybrid: ORB features for coarse alignment, ECC for sub-pixel refinement.
        """
        ref_u8 = self._to_gray_u8(self._get_ref_gray(ref_im))
        align_u8 = self._to_gray_u8(im_to_align)

        # Stage 1: Feature-based coarse alignment
        coarse_warp = self._feature_match(ref_u8, align_u8, mode="similarity")
        if coarse_warp is None:
            # Fallback to euclidean ECC
            return self._align_rst(ref_im, im_to_align, scale_factor)

        # Stage 2: Pre-warp and refine with ECC
        h, w = align_u8.shape[:2]
        prealigned = cv2.warpAffine(
            align_u8, coarse_warp, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residual = self._ecc_refine(ref_u8, prealigned, cv2.MOTION_EUCLIDEAN)

        # Stage 3: Compose coarse + residual
        C = np.vstack([coarse_warp, [0, 0, 1]])
        R = np.vstack([residual, [0, 0, 1]])
        combined = (R @ C)[:2, :].astype(np.float32)

        # Apply to full-res color image
        h, w = im_to_align.shape[:2]
        result = cv2.warpAffine(
            im_to_align, combined, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        self.last_alignment_mask = self._valid_mask_for_warp(combined, im_to_align.shape)
        self._track_warp_shifts(combined, im_to_align.shape)
        return result

    def _align_affine(self, ref_im, im_to_align, scale_factor):
        """
        6 DOF affine alignment: tx, ty, rotation, scale_x, scale_y, shear.
        Hybrid: ORB features for coarse alignment, ECC MOTION_AFFINE for refinement.
        """
        ref_u8 = self._to_gray_u8(self._get_ref_gray(ref_im))
        align_u8 = self._to_gray_u8(im_to_align)

        # Stage 1: Feature-based coarse alignment (full 6 DOF)
        coarse_warp = self._feature_match(ref_u8, align_u8, mode="affine")
        if coarse_warp is None:
            # Fallback to similarity
            return self._align_similarity(ref_im, im_to_align, scale_factor)

        # Stage 2: Pre-warp and refine with MOTION_AFFINE ECC
        h, w = align_u8.shape[:2]
        prealigned = cv2.warpAffine(
            align_u8, coarse_warp, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residual = self._ecc_refine(ref_u8, prealigned, cv2.MOTION_AFFINE)

        # Stage 3: Compose
        C = np.vstack([coarse_warp, [0, 0, 1]])
        R = np.vstack([residual, [0, 0, 1]])
        combined = (R @ C)[:2, :].astype(np.float32)

        h, w = im_to_align.shape[:2]
        result = cv2.warpAffine(
            im_to_align, combined, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        self.last_alignment_mask = self._valid_mask_for_warp(combined, im_to_align.shape)
        self._track_warp_shifts(combined, im_to_align.shape)
        return result

    def generate_laplacian_pyramid(self, im1, num_levels):
        if isinstance(im1, str):
            im1 = self.load_image(im1)
        if self.useGpu and HAS_GPU:
            return GPU.generate_laplacian_pyramid(im1, num_levels)
        return CPU.generate_laplacian_pyramid(im1, num_levels)

    def reconstruct_pyramid(self, laplacian_pyr):
        if self.useGpu and HAS_GPU:
            return GPU.reconstruct_pyramid(laplacian_pyr)
        return CPU.reconstruct_pyramid(laplacian_pyr)

    def focus_fuse_pyramid_pair(self, pyr1, pyr2, kernel_size,
                               contrast_threshold=0.0, feather_radius=0):
        # GPU path doesn't support contrast_threshold/feather_radius —
        # fall back to CPU when those are enabled for consistent results
        if self.useGpu and HAS_GPU and contrast_threshold <= 0 and feather_radius <= 0:
            return self._fuse_gpu(pyr1, pyr2, kernel_size)
        return self._fuse_cpu(pyr1, pyr2, kernel_size, contrast_threshold, feather_radius)

    def _fuse_cpu(self, pyr1, pyr2, kernel_size, contrast_threshold=0.0, feather_radius=0):
        threshold_index = len(pyr1) - 1
        new_pyr = []
        current_focusmap = None
        use_soft = feather_radius > 0

        for pyramid_level in range(len(pyr1)):
            if pyramid_level < threshold_index:
                if contrast_threshold > 0:
                    # Thresholded focusmap needs the Numba path
                    gray1 = cv2.cvtColor(pyr1[pyramid_level], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(pyr2[pyramid_level], cv2.COLOR_BGR2GRAY)
                    current_focusmap = CPU.compute_focusmap_thresholded(
                        gray1, gray2, kernel_size, np.float32(contrast_threshold),
                    )
                elif HAS_GPU:
                    # Use fast vectorized focusmap (cv2.blur variance, O(1)/pixel)
                    current_focusmap = GPU.compute_focusmap_fast(
                        pyr1[pyramid_level], pyr2[pyramid_level], kernel_size,
                    )
                else:
                    gray1 = cv2.cvtColor(pyr1[pyramid_level], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(pyr2[pyramid_level], cv2.COLOR_BGR2GRAY)
                    current_focusmap = CPU.compute_focusmap(
                        gray1, gray2, kernel_size,
                    )
            else:
                s = pyr2[pyramid_level].shape
                current_focusmap = cv2.resize(
                    current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
                )

            if use_soft:
                soft_map = CPU.feather_focusmap(current_focusmap, feather_radius)
                new_pyr_level = CPU.fuse_pyramid_levels_soft(
                    pyr1[pyramid_level], pyr2[pyramid_level], soft_map,
                )
            else:
                new_pyr_level = CPU.fuse_pyramid_levels_using_focusmap(
                    pyr1[pyramid_level], pyr2[pyramid_level], current_focusmap,
                )
            new_pyr.append(new_pyr_level)
        return new_pyr

    def _fuse_gpu(self, pyr1, pyr2, kernel_size):
        """GPU fusion — batched pipeline that keeps data on-GPU.
        Batch-uploads all pyramid levels, runs all kernels without
        intermediate host↔device transfers, downloads only at the end."""
        return GPU.fuse_pyramid_pair_gpu(pyr1, pyr2, kernel_size)
