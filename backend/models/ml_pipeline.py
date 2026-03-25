"""
ML Pipeline orchestrator.

Chains the three processing stages:
  Stage 1: Denoising
  Stage 2: Sharpening
  Stage 3: Color Grading

Reports per-stage timing and handles errors with fallback to previous stage output.
"""
import logging
import time
from typing import List

import numpy as np

from backend.models.denoiser import Denoiser
from backend.models.sharpener import Sharpener
from backend.models.color_grader import ColorGrader
from backend.models.profiles import get_profile, ProfileParams
from backend.api.schemas import StageTime

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Three-stage ML pipeline for RAW photo enhancement.

    Stages:
        1. Denoise: reduce sensor noise while preserving detail
        2. Sharpen: recover fine detail and add micro-contrast
        3. Color Grade: apply artistic color profile
    """

    def __init__(self, mode: str = "opencv"):
        self.mode = mode
        self.denoiser = None
        self.sharpener = None
        self.color_grader = None
        self._loaded = False

    def load(self):
        """Initialize all three pipeline stages."""
        logger.info(f"Loading pipeline stages (mode={self.mode})")

        self.denoiser = Denoiser(mode=self.mode)
        self.sharpener = Sharpener(mode=self.mode)
        self.color_grader = ColorGrader(mode=self.mode)
        self._loaded = True

        logger.info("All pipeline stages loaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def process(
        self,
        img: np.ndarray,
        profile_id: str,
        denoise_strength: float = 1.0,
        sharpen_strength: float = 1.0,
    ) -> tuple:
        """
        Run the full enhancement pipeline.

        Args:
            img: float32 RGB image [0, 1], shape (H, W, 3).
            profile_id: Name of the enhancement profile.
            denoise_strength: Denoising intensity (0.0–1.0).
            sharpen_strength: Sharpening intensity (0.5–2.0).

        Returns:
            Tuple of (enhanced_image, list_of_stage_times).
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        # Validate input
        assert img.ndim == 3 and img.shape[2] == 3, f"Expected (H,W,3), got {img.shape}"
        assert img.dtype == np.float32, f"Expected float32, got {img.dtype}"

        profile = get_profile(profile_id)
        stages: List[StageTime] = []
        current = img.copy()

        # Stage 1: Denoise
        t0 = time.time()
        try:
            current = self.denoiser.denoise(current, strength=denoise_strength)
            current = np.clip(current, 0.0, 1.0)
            dt = (time.time() - t0) * 1000
            stages.append(StageTime(name="Denoising", duration_ms=round(dt, 1)))
            logger.info(f"Stage 1 (Denoise): {dt:.1f}ms")
        except Exception as e:
            logger.error(f"Denoising failed: {e}, using original")
            stages.append(StageTime(name="Denoising", duration_ms=0))

        # Stage 2: Sharpen
        t0 = time.time()
        try:
            current = self.sharpener.sharpen(current, strength=sharpen_strength)
            current = np.clip(current, 0.0, 1.0)
            dt = (time.time() - t0) * 1000
            stages.append(StageTime(name="Sharpening", duration_ms=round(dt, 1)))
            logger.info(f"Stage 2 (Sharpen): {dt:.1f}ms")
        except Exception as e:
            logger.error(f"Sharpening failed: {e}, using denoised output")
            stages.append(StageTime(name="Sharpening", duration_ms=0))

        # Stage 3: Color Grade
        t0 = time.time()
        try:
            current = self.color_grader.grade(current, profile)
            current = np.clip(current, 0.0, 1.0)
            dt = (time.time() - t0) * 1000
            stages.append(StageTime(name="Color Grading", duration_ms=round(dt, 1)))
            logger.info(f"Stage 3 (Color Grade: {profile_id}): {dt:.1f}ms")
        except Exception as e:
            logger.error(f"Color grading failed: {e}, using sharpened output")
            stages.append(StageTime(name="Color Grading", duration_ms=0))

        return current, stages
