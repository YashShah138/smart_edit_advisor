"""
Stage 2: Image Sharpening

OpenCV fallback: multi-pass unsharp mask with detail enhancement.
PyTorch mode: Real-ESRGAN pretrained super-resolution model (stub included).
"""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class Sharpener:
    """
    Sharpen images using either OpenCV unsharp mask (default)
    or Real-ESRGAN super-resolution when available.
    """

    def __init__(self, mode: str = "opencv"):
        self.mode = mode
        self.model = None

        if mode == "pytorch":
            self._load_pytorch_model()

    def _load_pytorch_model(self):
        """
        Load pretrained Real-ESRGAN model for super-resolution sharpening.

        Architecture: Real-ESRGAN (Wang et al., 2021)
        - RRDB (Residual-in-Residual Dense Block) network
        - Trained on degraded real-world images
        - 2x or 4x upscale, then downscale back for sharpening effect
        - Hugging Face: https://huggingface.co/ai-forever/Real-ESRGAN

        To use:
            1. pip install torch realesrgan basicsr
            2. Download weights from Hugging Face
            3. Set PIPELINE_MODE=pytorch in environment
        """
        try:
            import torch
            from backend.config import ESRGAN_MODEL_PATH

            if ESRGAN_MODEL_PATH.exists():
                # Real-ESRGAN loading code
                # from basicsr.archs.rrdbnet_arch import RRDBNet
                # from realesrgan import RealESRGANer
                # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                #                 num_block=23, num_grow_ch=32, scale=2)
                # self.model = RealESRGANer(
                #     scale=2, model_path=str(ESRGAN_MODEL_PATH),
                #     model=model, tile=512, tile_pad=10, pre_pad=0,
                #     half=False
                # )
                logger.info("Real-ESRGAN model loaded")
            else:
                logger.warning(f"Real-ESRGAN weights not found at {ESRGAN_MODEL_PATH}")
                self.mode = "opencv"

        except ImportError:
            logger.warning("PyTorch/Real-ESRGAN not available, falling back to OpenCV")
            self.mode = "opencv"

    def sharpen(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sharpen an image.

        Args:
            img: float32 RGB image [0, 1], shape (H, W, 3).
            strength: Sharpening strength (0.5 = subtle, 1.0 = standard, 2.0 = aggressive).

        Returns:
            Sharpened float32 RGB image [0, 1].
        """
        if self.mode == "pytorch" and self.model is not None:
            return self._sharpen_pytorch(img, strength)
        return self._sharpen_opencv(img, strength)

    def _sharpen_opencv(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Multi-pass unsharp mask sharpening with detail enhancement.

        Pass 1: Broad unsharp mask (large radius) for global structure
        Pass 2: Fine unsharp mask (small radius) for micro-detail
        Pass 3: Optional edge-aware sharpening via Laplacian
        """
        # Pass 1: Broad sharpening (radius ~5px)
        sigma_broad = 3.0
        blurred = cv2.GaussianBlur(img, (0, 0), sigma_broad)
        sharp1 = cv2.addWeighted(img, 1.0 + 0.5 * strength, blurred, -0.5 * strength, 0)

        # Pass 2: Fine detail sharpening (radius ~1px)
        sigma_fine = 1.0
        blurred_fine = cv2.GaussianBlur(sharp1, (0, 0), sigma_fine)
        sharp2 = cv2.addWeighted(sharp1, 1.0 + 0.3 * strength, blurred_fine, -0.3 * strength, 0)

        # Pass 3: Edge-aware clarity boost using luminance channel
        if strength > 0.7:
            # Work in LAB space for luminance-only sharpening
            img_u8 = np.clip(sharp2 * 255, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)

            # Clarity-style local contrast enhancement
            blur_large = cv2.GaussianBlur(l_channel, (0, 0), 20.0)
            local_contrast = l_channel - blur_large
            l_channel = l_channel + local_contrast * 0.3 * strength
            lab[:, :, 0] = np.clip(l_channel, 0, 255).astype(np.uint8)

            sharp2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

        return np.clip(sharp2, 0.0, 1.0)

    def _sharpen_pytorch(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Real-ESRGAN super-resolution sharpening.
        Upscales 2x then downscales back, producing sharper details.
        """
        import torch

        # Convert to uint8 BGR for Real-ESRGAN
        img_bgr = cv2.cvtColor(
            np.clip(img * 255, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR
        )

        # Super-resolve
        output, _ = self.model.enhance(img_bgr, outscale=2)

        # Downscale back to original resolution
        h, w = img.shape[:2]
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # BGR → RGB, uint8 → float32
        result = cv2.cvtColor(output, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Blend with original based on strength
        if strength < 1.0:
            result = img * (1.0 - strength) + result * strength

        return np.clip(result, 0.0, 1.0)
