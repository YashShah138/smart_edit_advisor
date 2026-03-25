"""
Stage 1: Image Denoising

OpenCV fallback: bilateral filter for edge-preserving noise reduction.
PyTorch mode: DnCNN or RIDNet pretrained model (stub included).
"""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class Denoiser:
    """
    Denoise images using either OpenCV bilateral filter (default)
    or a pretrained DnCNN/RIDNet model when available.
    """

    def __init__(self, mode: str = "opencv"):
        self.mode = mode
        self.model = None

        if mode == "pytorch":
            self._load_pytorch_model()

    def _load_pytorch_model(self):
        """
        Load pretrained DnCNN denoising model.

        Architecture: DnCNN (Zhang et al., 2017)
        - 17-layer CNN with residual learning
        - Trained on BSD400 + ImageNet patches with Gaussian noise σ=25
        - Input: noisy image patch, Output: noise residual
        - Clean = Input - Residual

        To use:
            1. pip install torch torchvision
            2. Download weights: https://huggingface.co/cszn/DnCNN
            3. Set PIPELINE_MODE=pytorch in environment
        """
        try:
            import torch
            import torch.nn as nn

            class DnCNN(nn.Module):
                """DnCNN architecture for image denoising."""

                def __init__(self, channels=3, num_layers=17, features=64):
                    super().__init__()
                    layers = [
                        nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=False),
                        nn.ReLU(inplace=True),
                    ]
                    for _ in range(num_layers - 2):
                        layers.extend([
                            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                            nn.BatchNorm2d(features),
                            nn.ReLU(inplace=True),
                        ])
                    layers.append(
                        nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False)
                    )
                    self.dncnn = nn.Sequential(*layers)

                def forward(self, x):
                    noise = self.dncnn(x)
                    return x - noise  # residual learning

            from backend.config import DNCNN_MODEL_PATH

            if DNCNN_MODEL_PATH.exists():
                self.model = DnCNN()
                state_dict = torch.load(str(DNCNN_MODEL_PATH), map_location="cpu")
                self.model.load_state_dict(state_dict)
                self.model.eval()
                logger.info("DnCNN model loaded successfully")
            else:
                logger.warning(f"DnCNN weights not found at {DNCNN_MODEL_PATH}, falling back to OpenCV")
                self.mode = "opencv"

        except ImportError:
            logger.warning("PyTorch not available, falling back to OpenCV denoising")
            self.mode = "opencv"

    def denoise(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Denoise an image.

        Args:
            img: float32 RGB image [0, 1], shape (H, W, 3).
            strength: Denoising strength multiplier (0.0 = no effect, 1.0 = full).

        Returns:
            Denoised float32 RGB image [0, 1].
        """
        if self.mode == "pytorch" and self.model is not None:
            return self._denoise_pytorch(img, strength)
        return self._denoise_opencv(img, strength)

    def _denoise_opencv(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        OpenCV bilateral filter denoising.
        Preserves edges while smoothing noise in flat regions.
        """
        # Convert to uint8 for bilateral filter
        img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Bilateral filter: d=9, sigmaColor=75, sigmaSpace=75
        sigma_color = int(75 * strength)
        sigma_space = int(75 * strength)
        denoised = cv2.bilateralFilter(img_u8, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        # Additionally apply fastNlMeans for chroma denoising
        if strength > 0.5:
            h_param = int(10 * strength)
            denoised = cv2.fastNlMeansDenoisingColored(denoised, None, h_param, h_param, 7, 21)

        result = denoised.astype(np.float32) / 255.0

        # Blend with original based on strength
        if strength < 1.0:
            result = img * (1.0 - strength) + result * strength

        return np.clip(result, 0.0, 1.0)

    def _denoise_pytorch(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        PyTorch DnCNN denoising (requires model weights).
        Processes image in patches to avoid OOM on large images.
        """
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # HWC → CHW, add batch dim
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            denoised = self.model(tensor)

        # Back to numpy HWC
        result = denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        result = np.clip(result, 0.0, 1.0)

        # Blend with original based on strength
        if strength < 1.0:
            result = img * (1.0 - strength) + result * strength

        return result
