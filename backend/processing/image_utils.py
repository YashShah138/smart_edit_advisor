"""
Image processing utility functions.
"""
import io
import base64
import logging
from typing import Tuple

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


def normalize(img: np.ndarray) -> np.ndarray:
    """Ensure image is float32 in [0, 1] range."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.dtype == np.float32:
        return np.clip(img, 0.0, 1.0)
    else:
        return img.astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] image to uint8 [0,255]."""
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] image to uint16 [0,65535]."""
    return np.clip(img * 65535.0, 0, 65535).astype(np.uint16)


def encode_jpeg_base64(img: np.ndarray, quality: int = 95) -> str:
    """
    Encode a float32 RGB image as a base64 JPEG string.

    Args:
        img: float32 [0,1] RGB image.
        quality: JPEG quality (1-100).

    Returns:
        Base64 encoded JPEG string with data URI prefix.
    """
    rgb_uint8 = to_uint8(img)
    pil_img = Image.fromarray(rgb_uint8, mode="RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def save_jpeg(img: np.ndarray, path: str, quality: int = 95) -> None:
    """Save float32 RGB image as JPEG file."""
    rgb_uint8 = to_uint8(img)
    pil_img = Image.fromarray(rgb_uint8, mode="RGB")
    pil_img.save(path, format="JPEG", quality=quality)
    logger.info(f"Saved JPEG: {path}")


def resize_to_fit(img: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Resize image so its largest dimension is at most max_dim pixels.
    Maintains aspect ratio.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_thumbnail(img: np.ndarray, size: int = 400) -> np.ndarray:
    """Create a square-ish thumbnail for preview."""
    return resize_to_fit(img, size)
