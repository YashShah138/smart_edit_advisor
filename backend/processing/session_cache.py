"""
Session-based cache for demosaiced RAW images.

Stores decoded RAW arrays to disk so users can switch enhancement profiles
without re-decoding the RAW file each time.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import MAX_CACHED_SESSIONS, CACHE_TTL_HOURS

# Use /tmp for cache to avoid permission issues on mounted filesystems
CACHE_DIR = Path("/tmp/raw-enhance-cache")

logger = logging.getLogger(__name__)


class SessionCache:
    """
    File-backed cache for demosaiced RAW image arrays.

    Cache key: SHA256 hash of the uploaded file content.
    Storage: numpy .npy files in the cache directory.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_key(self, file_content: bytes) -> str:
        """Compute cache key from file content hash."""
        return hashlib.sha256(file_content).hexdigest()[:16]

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve a cached demosaiced image.

        Args:
            key: Cache key (file content hash).

        Returns:
            float32 RGB array or None if not cached.
        """
        npy_path = self.cache_dir / f"{key}.npy"
        meta_path = self.cache_dir / f"{key}.json"

        if not npy_path.exists():
            return None

        # Check TTL
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                cached_at = meta.get("cached_at", 0)
                if time.time() - cached_at > CACHE_TTL_HOURS * 3600:
                    logger.info(f"Cache expired for {key}")
                    self._remove(key)
                    return None
            except (json.JSONDecodeError, KeyError):
                pass

        try:
            img = np.load(str(npy_path), allow_pickle=False)
            logger.info(f"Cache hit: {key} (shape={img.shape})")
            return img
        except Exception as e:
            logger.warning(f"Failed to load cache {key}: {e}")
            self._remove(key)
            return None

    def put(self, key: str, img: np.ndarray, metadata: dict = None) -> None:
        """
        Store a demosaiced image in the cache.

        Args:
            key: Cache key.
            img: float32 RGB array to cache.
            metadata: Optional metadata dict (file format, dimensions, etc.).
        """
        # Enforce cache size limit
        self._evict_if_needed()

        npy_path = self.cache_dir / f"{key}.npy"
        meta_path = self.cache_dir / f"{key}.json"

        try:
            np.save(str(npy_path), img)

            meta = {
                "cached_at": time.time(),
                "shape": list(img.shape),
                "dtype": str(img.dtype),
                **(metadata or {}),
            }
            meta_path.write_text(json.dumps(meta))

            logger.info(f"Cached: {key} (shape={img.shape}, size={npy_path.stat().st_size / 1e6:.1f}MB)")
        except Exception as e:
            logger.error(f"Failed to cache {key}: {e}")
            self._remove(key)

    def _remove(self, key: str) -> None:
        """Remove a cache entry."""
        for ext in [".npy", ".json"]:
            p = self.cache_dir / f"{key}{ext}"
            if p.exists():
                p.unlink()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max size."""
        entries = list(self.cache_dir.glob("*.json"))
        if len(entries) <= MAX_CACHED_SESSIONS:
            return

        # Sort by cached_at, evict oldest
        timed = []
        for meta_path in entries:
            try:
                meta = json.loads(meta_path.read_text())
                timed.append((meta.get("cached_at", 0), meta_path.stem))
            except Exception:
                timed.append((0, meta_path.stem))

        timed.sort()
        to_remove = len(timed) - MAX_CACHED_SESSIONS + 5  # remove a few extra
        for _, key in timed[:to_remove]:
            self._remove(key)
            logger.info(f"Evicted cache entry: {key}")

    def clear_all(self) -> None:
        """Remove all cached entries."""
        for f in self.cache_dir.iterdir():
            f.unlink()
        logger.info("Cleared all cache entries")
