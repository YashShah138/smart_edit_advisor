"""
Stage 3: Color Grading

Applies parametric color adjustments based on enhancement profiles.
OpenCV fallback: curve-based grading with shadows/midtones/highlights control.
PyTorch mode: lightweight CNN trained on DPED phone→DSLR pairs (stub included).
"""
import logging
from typing import List, Tuple

import numpy as np
import cv2

from backend.models.profiles import ProfileParams, ToneCurve

logger = logging.getLogger(__name__)


class ColorGrader:
    """
    Apply color grading to images based on profile parameters.

    The grading pipeline applies adjustments in this order:
    1. Exposure compensation
    2. White balance / temperature shift
    3. Tone curve (luminance)
    4. Per-channel curves (RGB)
    5. Contrast adjustment
    6. Saturation / vibrance
    7. Shadow lift / highlight roll-off
    8. Split-toning
    9. Clarity (local contrast)
    10. B&W conversion (if profile is monochrome)
    11. Fade (lifted blacks)
    12. Film grain (optional)
    """

    def __init__(self, mode: str = "opencv"):
        self.mode = mode

        if mode == "pytorch":
            self._load_pytorch_model()

    def _load_pytorch_model(self):
        """
        Load lightweight color grading CNN trained on the DPED dataset.

        Architecture options:
        - 3DLUT-based network (Zeng et al., 2020): learns a 3D lookup table
        - Small UNet: 4 encoder + 4 decoder blocks, ~2M parameters
        - Trained on DPED: ~6000 phone→DSLR paired images per device

        Training data:
            Input: smartphone JPEG (iPhone / BlackBerry / Sony)
            Target: Canon DSLR JPEG (same scene)
            See backend/training/train.py for training script.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            # Must match the ColorGradeUNet architecture in backend/training/train.py exactly.
            class ConvBlock(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    return self.conv(x)

            class ColorGradeUNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.enc1 = ConvBlock(3, 32)
                    self.enc2 = ConvBlock(32, 64)
                    self.enc3 = ConvBlock(64, 128)
                    self.enc4 = ConvBlock(128, 256)
                    self.pool = nn.MaxPool2d(2)
                    self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
                    self.dec3 = ConvBlock(256, 128)
                    self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
                    self.dec2 = ConvBlock(128, 64)
                    self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
                    self.dec1 = ConvBlock(64, 32)
                    self.final = nn.Sequential(nn.Conv2d(32, 3, 1), nn.Sigmoid())

                def _cat(self, up, skip):
                    return torch.cat([up, skip[:, :, :up.shape[2], :up.shape[3]]], dim=1)

                def forward(self, x):
                    h, w = x.shape[2], x.shape[3]
                    pad_h = (8 - h % 8) % 8
                    pad_w = (8 - w % 8) % 8
                    if pad_h or pad_w:
                        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool(e1))
                    e3 = self.enc3(self.pool(e2))
                    e4 = self.enc4(self.pool(e3))
                    d3 = self.dec3(self._cat(self.up3(e4), e3))
                    d2 = self.dec2(self._cat(self.up2(d3), e2))
                    d1 = self.dec1(self._cat(self.up1(d2), e1))
                    return self.final(d1)[:, :, :h, :w]

            from backend.config import COLORGRADE_MODEL_PATH
            if COLORGRADE_MODEL_PATH.exists():
                self.model = ColorGradeUNet()
                state_dict = torch.load(str(COLORGRADE_MODEL_PATH), map_location="cpu")
                self.model.load_state_dict(state_dict)
                self.model.eval()
                logger.info("Color grading UNet loaded from %s", COLORGRADE_MODEL_PATH)
            else:
                logger.warning("Color grading model not found at %s, using parametric fallback", COLORGRADE_MODEL_PATH)
                self.mode = "opencv"
        except ImportError:
            logger.warning("PyTorch not available for color grading, using parametric mode")
            self.mode = "opencv"

    def grade(self, img: np.ndarray, profile: ProfileParams) -> np.ndarray:
        """
        Apply color grading to an image.

        Args:
            img: float32 RGB image [0, 1], shape (H, W, 3).
            profile: ProfileParams with grading parameters.

        Returns:
            Graded float32 RGB image [0, 1].
        """
        result = img.copy()

        # 1. Exposure
        if profile.exposure != 0:
            result = self._apply_exposure(result, profile.exposure)

        # 2. Temperature / white balance
        if profile.temperature != 0 or profile.tint != 0:
            result = self._apply_temperature(result, profile.temperature, profile.tint)

        # 3. Master tone curve
        if len(profile.tone_curve.points) > 2:
            result = self._apply_tone_curve(result, profile.tone_curve)

        # 4. Per-channel curves
        for ch, curve in enumerate([profile.red_curve, profile.green_curve, profile.blue_curve]):
            if len(curve.points) > 2:
                result[:, :, ch] = self._interpolate_curve(result[:, :, ch], curve.points)

        # 5. Contrast
        if profile.contrast != 0:
            result = self._apply_contrast(result, profile.contrast)

        # 6. Saturation & vibrance
        if profile.saturation != 0:
            result = self._apply_saturation(result, profile.saturation)
        if profile.vibrance != 0:
            result = self._apply_vibrance(result, profile.vibrance)

        # 7. Shadow lift & highlight roll-off
        if profile.shadow_lift > 0:
            result = self._lift_shadows(result, profile.shadow_lift)
        if profile.highlight_rolloff > 0:
            result = self._rolloff_highlights(result, profile.highlight_rolloff)

        # 8. Split toning
        if profile.shadow_tint.intensity > 0:
            result = self._apply_split_tone(result, profile.shadow_tint, shadows=True)
        if profile.highlight_tint.intensity > 0:
            result = self._apply_split_tone(result, profile.highlight_tint, shadows=False)

        # 9. Clarity
        if profile.clarity != 0:
            result = self._apply_clarity(result, profile.clarity)

        # 10. B&W conversion
        if profile.is_bw:
            result = self._convert_bw(
                result, profile.bw_red_weight, profile.bw_green_weight, profile.bw_blue_weight
            )

        # 11. Fade (lifted blacks)
        if profile.fade > 0:
            result = result * (1.0 - profile.fade) + profile.fade

        # 12. Film grain
        if profile.grain > 0:
            result = self._add_grain(result, profile.grain)

        return np.clip(result, 0.0, 1.0)

    # ── Internal adjustment methods ───────────────────────────────────────

    def _apply_exposure(self, img: np.ndarray, ev: float) -> np.ndarray:
        """Apply exposure compensation in EV stops."""
        return np.clip(img * (2.0 ** ev), 0.0, 1.0)

    def _apply_temperature(self, img: np.ndarray, temp: float, tint: float) -> np.ndarray:
        """
        Shift color temperature (warm/cool) and tint (green/magenta).
        temp > 0 = warmer, temp < 0 = cooler.
        """
        result = img.copy()
        # Warm: boost red, reduce blue
        temp_factor = temp / 100.0
        result[:, :, 0] = np.clip(result[:, :, 0] + temp_factor * 0.1, 0, 1)  # Red
        result[:, :, 2] = np.clip(result[:, :, 2] - temp_factor * 0.1, 0, 1)  # Blue

        # Tint: green/magenta
        tint_factor = tint / 100.0
        result[:, :, 1] = np.clip(result[:, :, 1] + tint_factor * 0.05, 0, 1)  # Green

        return result

    def _apply_tone_curve(self, img: np.ndarray, curve: ToneCurve) -> np.ndarray:
        """Apply a tone curve to image luminance."""
        # Convert to LAB, apply curve to L channel, convert back
        img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Normalize L channel to [0, 1]
        l_norm = lab[:, :, 0] / 255.0

        # Apply curve
        l_norm = self._interpolate_curve(l_norm, curve.points)

        # Back to LAB range
        lab[:, :, 0] = np.clip(l_norm * 255, 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

        return result

    def _interpolate_curve(self, channel: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Apply a piecewise-linear curve to a single channel.
        Points: list of (input, output) pairs in [0, 1].
        """
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        # Build a 256-entry LUT for fast application
        lut_x = np.linspace(0, 1, 256)
        lut_y = np.interp(lut_x, xs, ys)

        # Quantize input to LUT indices and look up
        indices = np.clip(channel * 255, 0, 255).astype(np.int32)
        return lut_y[indices].astype(np.float32)

    def _apply_contrast(self, img: np.ndarray, amount: float) -> np.ndarray:
        """
        S-curve contrast adjustment.
        amount > 0 = more contrast, amount < 0 = less contrast.
        """
        midpoint = 0.5
        # Sigmoidal contrast
        factor = 1.0 + amount
        result = midpoint + (img - midpoint) * factor
        return np.clip(result, 0.0, 1.0)

    def _apply_saturation(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Adjust global saturation."""
        gray = np.mean(img, axis=2, keepdims=True)
        factor = 1.0 + amount
        result = gray + (img - gray) * factor
        return np.clip(result, 0.0, 1.0)

    def _apply_vibrance(self, img: np.ndarray, amount: float) -> np.ndarray:
        """
        Vibrance: saturation boost that's stronger on less-saturated pixels.
        Protects skin tones and already-saturated colors.
        """
        # Compute per-pixel saturation (max - min across channels)
        sat = np.max(img, axis=2) - np.min(img, axis=2)
        # Invert: low saturation → high boost factor
        boost_mask = (1.0 - sat) ** 2

        gray = np.mean(img, axis=2, keepdims=True)
        factor = 1.0 + amount * boost_mask[:, :, np.newaxis]
        result = gray + (img - gray) * factor
        return np.clip(result, 0.0, 1.0)

    def _lift_shadows(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Lift shadow tones (raise the black point)."""
        return img * (1.0 - amount) + amount

    def _rolloff_highlights(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Roll off highlights (compress bright values)."""
        return img * (1.0 - amount)

    def _apply_split_tone(self, img: np.ndarray, tint, shadows: bool) -> np.ndarray:
        """Apply color tint to shadows or highlights."""
        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        if shadows:
            mask = (1.0 - luminance) ** 2  # Stronger in dark areas
        else:
            mask = luminance ** 2  # Stronger in bright areas

        mask = mask[:, :, np.newaxis] * tint.intensity

        tint_color = np.array([tint.r, tint.g, tint.b], dtype=np.float32)
        result = img + mask * tint_color

        return np.clip(result, 0.0, 1.0)

    def _apply_clarity(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Local contrast enhancement (clarity slider)."""
        img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

        l_channel = lab[:, :, 0]
        # Large-radius blur for local average
        blur = cv2.GaussianBlur(l_channel, (0, 0), 20.0)
        local_contrast = l_channel - blur
        l_channel = l_channel + local_contrast * amount
        lab[:, :, 0] = np.clip(l_channel, 0, 255)

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return result

    def _convert_bw(self, img: np.ndarray, r_w: float, g_w: float, b_w: float) -> np.ndarray:
        """Weighted luminosity-based B&W conversion."""
        mono = img[:, :, 0] * r_w + img[:, :, 1] * g_w + img[:, :, 2] * b_w
        return np.stack([mono, mono, mono], axis=2)

    def _add_grain(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Add subtle film grain noise."""
        noise = np.random.normal(0, 0.02 * amount, img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0)
