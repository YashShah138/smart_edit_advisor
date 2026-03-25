"""
Patch-based image processing for large RAW files.

Splits images into overlapping patches, processes each through the ML pipeline,
and blends them back with linear fade at seam boundaries.
"""
import logging
from typing import Callable

import numpy as np

from backend.config import PATCH_SIZE, PATCH_OVERLAP, LARGE_IMAGE_THRESHOLD

logger = logging.getLogger(__name__)


class PatchProcessor:
    """
    Process large images in overlapping patches to avoid OOM errors.

    Uses linear blending in overlap regions for seamless stitching.
    """

    def __init__(self, patch_size: int = PATCH_SIZE, overlap: int = PATCH_OVERLAP):
        self.patch_size = patch_size
        self.overlap = overlap

    def needs_patching(self, img: np.ndarray) -> bool:
        """Check if image is large enough to require patch processing."""
        h, w = img.shape[:2]
        return max(h, w) > LARGE_IMAGE_THRESHOLD

    def process(
        self,
        img: np.ndarray,
        process_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Process an image through patches if needed, or full-image otherwise.

        Args:
            img: float32 RGB [0,1], shape (H, W, 3).
            process_fn: Function that takes a patch and returns processed patch.

        Returns:
            Processed image, same shape as input.
        """
        if not self.needs_patching(img):
            logger.info("Image small enough for full processing")
            return process_fn(img)

        h, w, c = img.shape
        step = self.patch_size - self.overlap
        logger.info(
            f"Patch processing: img={w}x{h}, patch={self.patch_size}, "
            f"overlap={self.overlap}, step={step}"
        )

        # Output accumulator and weight map for blending
        output = np.zeros_like(img)
        weight_map = np.zeros((h, w, 1), dtype=np.float32)

        # Create blending weight for a single patch (linear ramp at edges)
        patch_weight = self._create_blend_weight(self.patch_size, self.patch_size, self.overlap)

        patch_count = 0
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Extract patch (handle edge cases)
                y_end = min(y + self.patch_size, h)
                x_end = min(x + self.patch_size, w)
                y_start = max(0, y_end - self.patch_size)
                x_start = max(0, x_end - self.patch_size)

                patch = img[y_start:y_end, x_start:x_end]
                ph, pw = patch.shape[:2]

                # Process patch
                processed = process_fn(patch)

                # Create weight for this patch size (may differ at edges)
                if ph != self.patch_size or pw != self.patch_size:
                    w_patch = self._create_blend_weight(ph, pw, self.overlap)
                else:
                    w_patch = patch_weight[:ph, :pw]

                # Accumulate
                output[y_start:y_end, x_start:x_end] += processed * w_patch
                weight_map[y_start:y_end, x_start:x_end] += w_patch

                patch_count += 1

        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-8)  # avoid division by zero
        output = output / weight_map

        logger.info(f"Processed {patch_count} patches")
        return np.clip(output, 0.0, 1.0)

    def _create_blend_weight(self, h: int, w: int, overlap: int) -> np.ndarray:
        """
        Create a 2D weight map with linear ramps in the overlap region.
        Center of patch has weight 1.0, edges ramp down linearly.
        """
        # 1D ramp for each dimension
        wy = np.ones(h, dtype=np.float32)
        wx = np.ones(w, dtype=np.float32)

        if overlap > 0 and h > overlap:
            ramp = np.linspace(0, 1, overlap, dtype=np.float32)
            wy[:overlap] = ramp
            wy[-overlap:] = ramp[::-1]

        if overlap > 0 and w > overlap:
            ramp = np.linspace(0, 1, overlap, dtype=np.float32)
            wx[:overlap] = ramp
            wx[-overlap:] = ramp[::-1]

        # 2D weight = outer product
        weight = wy[:, np.newaxis] * wx[np.newaxis, :]
        return weight[:, :, np.newaxis]  # (H, W, 1) for broadcasting
