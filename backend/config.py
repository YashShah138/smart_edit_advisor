"""
Configuration for the RAW Photo Enhancement Pipeline.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
UPLOAD_DIR = BACKEND_DIR / "uploads"
CACHE_DIR = UPLOAD_DIR / "cache"
OUTPUT_DIR = UPLOAD_DIR / "outputs"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File upload constraints
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_RAW_EXTENSIONS = {".cr2", ".nef", ".arw", ".dng"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ALLOWED_EXTENSIONS = ALLOWED_RAW_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS

# Processing parameters
PATCH_SIZE = 2048
PATCH_OVERLAP = 256
LARGE_IMAGE_THRESHOLD = 4000  # pixels — trigger patch processing above this
OUTPUT_JPEG_QUALITY = 95

# Session cache
MAX_CACHED_SESSIONS = 50
CACHE_TTL_HOURS = 24

# ML pipeline mode: "opencv" for fallback, "pytorch" for real models
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "opencv")

# Model paths (for PyTorch mode)
DNCNN_MODEL_PATH = BACKEND_DIR / "weights" / "dncnn.pth"
ESRGAN_MODEL_PATH = BACKEND_DIR / "weights" / "realesrgan.pth"
COLORGRADE_MODEL_PATH = BACKEND_DIR / "weights" / "colorgrade.pth"

# Server
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
