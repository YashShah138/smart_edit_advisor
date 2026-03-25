"""
RAW file decoder using rawpy (LibRaw wrapper).
Handles .CR2, .NEF, .ARW, .DNG and fallback JPG/PNG loading.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

from backend.api.schemas import RawMetadata
from backend.config import ALLOWED_RAW_EXTENSIONS

logger = logging.getLogger(__name__)


class RawDecoder:
    """Decode RAW image files to linear RGB float32 arrays."""

    SUPPORTED_RAW = ALLOWED_RAW_EXTENSIONS

    def decode(self, file_path: str) -> np.ndarray:
        """
        Decode a RAW or standard image file to float32 RGB [0, 1].

        Args:
            file_path: Path to the image file.

        Returns:
            np.ndarray of shape (H, W, 3), dtype float32, range [0, 1].
        """
        ext = Path(file_path).suffix.lower()

        if ext in self.SUPPORTED_RAW:
            return self._decode_raw(file_path)
        else:
            return self._decode_standard(file_path)

    def _decode_raw(self, file_path: str) -> np.ndarray:
        """Decode RAW file using rawpy."""
        try:
            import rawpy
        except ImportError:
            logger.warning("rawpy not available, falling back to OpenCV loader")
            return self._decode_standard(file_path)

        logger.info(f"Decoding RAW file: {file_path}")
        with rawpy.imread(file_path) as raw:
            # Demosaic with default parameters — produces 16-bit linear RGB
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
                output_color=rawpy.ColorSpace.sRGB,
            )

        # Convert 16-bit uint16 → float32 [0, 1]
        img = rgb.astype(np.float32) / 65535.0
        logger.info(f"Decoded RAW: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")
        return img

    def _decode_standard(self, file_path: str) -> np.ndarray:
        """Decode JPG/PNG/TIFF via OpenCV."""
        logger.info(f"Decoding standard image: {file_path}")
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {file_path}")
        # BGR → RGB, uint8 → float32 [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        logger.info(f"Decoded image: shape={img.shape}")
        return img

    def extract_metadata(self, file_path: str, img: Optional[np.ndarray] = None) -> RawMetadata:
        """
        Extract EXIF metadata from a RAW or image file.

        Args:
            file_path: Path to the image file.
            img: Optional pre-decoded image array (for dimensions).

        Returns:
            RawMetadata with available fields populated.
        """
        p = Path(file_path)
        ext = p.suffix.lower()
        file_size_mb = round(p.stat().st_size / (1024 * 1024), 2)

        meta = RawMetadata(
            file_format=ext.lstrip(".").upper(),
            file_size_mb=file_size_mb,
        )

        # Get dimensions from image array if available
        if img is not None:
            meta.height, meta.width = img.shape[:2]

        # Try to extract EXIF from RAW files
        if ext in self.SUPPORTED_RAW:
            meta = self._extract_raw_exif(file_path, meta)
        else:
            meta = self._extract_standard_exif(file_path, meta)

        return meta

    def _extract_raw_exif(self, file_path: str, meta: RawMetadata) -> RawMetadata:
        """Extract EXIF from RAW using rawpy."""
        try:
            import rawpy
            with rawpy.imread(file_path) as raw:
                meta.width = raw.sizes.width
                meta.height = raw.sizes.height
                # rawpy doesn't expose full EXIF; we extract what we can
                meta.camera_model = getattr(raw, "camera_make", "") + " " + getattr(raw, "camera_model", "")
                if not meta.camera_model.strip():
                    meta.camera_model = "Unknown"
        except Exception as e:
            logger.warning(f"Could not extract RAW EXIF: {e}")

        return meta

    def _extract_standard_exif(self, file_path: str, meta: RawMetadata) -> RawMetadata:
        """Extract EXIF from JPG/PNG using Pillow."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            with Image.open(file_path) as pil_img:
                if meta.width == 0:
                    meta.width, meta.height = pil_img.size

                exif_data = pil_img.getexif()
                if exif_data:
                    tag_map = {TAGS.get(k, k): v for k, v in exif_data.items()}
                    meta.camera_model = str(tag_map.get("Model", "Unknown"))
                    meta.iso = tag_map.get("ISOSpeedRatings")

                    if "FNumber" in tag_map:
                        f = tag_map["FNumber"]
                        if hasattr(f, "numerator"):
                            meta.aperture = f"f/{f.numerator / f.denominator:.1f}"

                    if "ExposureTime" in tag_map:
                        et = tag_map["ExposureTime"]
                        if hasattr(et, "numerator") and et.numerator > 0:
                            if et.numerator == 1:
                                meta.shutter_speed = f"1/{et.denominator}s"
                            else:
                                meta.shutter_speed = f"{et.numerator}/{et.denominator}s"

                    if "FocalLength" in tag_map:
                        fl = tag_map["FocalLength"]
                        if hasattr(fl, "numerator"):
                            meta.focal_length = f"{fl.numerator / fl.denominator:.0f}mm"
        except Exception as e:
            logger.warning(f"Could not extract EXIF: {e}")

        return meta
