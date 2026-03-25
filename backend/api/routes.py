"""
API route definitions.

POST /enhance  — Upload RAW file + profile, return enhanced image
GET  /profiles — List available enhancement profiles
GET  /health   — Health check
"""
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from backend.config import (
    MAX_FILE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    UPLOAD_DIR,
    OUTPUT_DIR,
    OUTPUT_JPEG_QUALITY,
)
from backend.api.schemas import (
    EnhancementResponse,
    ProfileInfo,
    ProfileName,
    HealthResponse,
    StageTime,
)
from backend.models.raw_decoder import RawDecoder
from backend.models.profiles import PROFILE_INFO, PROFILES
from backend.processing.image_utils import encode_jpeg_base64, resize_to_fit
from backend.processing.session_cache import SessionCache
from backend.processing.patch_processor import PatchProcessor

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared instances
decoder = RawDecoder()
cache = SessionCache()
patch_processor = PatchProcessor()


def _get_pipeline():
    """Import pipeline from main module (avoids circular import)."""
    from backend.main import get_pipeline
    return get_pipeline()


@router.post("/enhance", response_model=EnhancementResponse)
async def enhance_image(
    file: UploadFile = File(...),
    profile: str = Form(default="expert_natural"),
):
    """
    Enhance a RAW or standard image file.

    1. Validate file size and format
    2. Decode RAW (or load standard image), cache result
    3. Run three-stage ML pipeline with selected profile
    4. Return base64-encoded before/after images + metadata
    """
    total_start = time.time()

    # ── Validate profile ─────────────────────────────────────────────
    if profile not in PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown profile '{profile}'. Available: {list(PROFILES.keys())}",
        )

    # ── Validate file ────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Accepted: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) / 1e6:.1f}MB). Maximum: {MAX_FILE_SIZE_BYTES / 1e6:.0f}MB",
        )

    # ── Decode / cache RAW ───────────────────────────────────────────
    import tempfile

    stages = []
    cache_key = cache.compute_key(content)

    # Check cache for demosaiced image
    img = cache.get(cache_key)

    if img is not None:
        stages.append(StageTime(name="Decoding (cached)", duration_ms=0))
    else:
        # Save temp file for decoder (use /tmp to avoid permission issues)
        temp_path = Path(tempfile.gettempdir()) / f"{cache_key}{ext}"
        temp_path.write_bytes(content)

        try:
            t0 = time.time()
            img = decoder.decode(str(temp_path))
            dt = (time.time() - t0) * 1000
            stages.append(StageTime(name="Decoding", duration_ms=round(dt, 1)))
            logger.info(f"Decoded in {dt:.1f}ms: {img.shape}")

            # Cache the decoded image
            cache.put(cache_key, img, {"format": ext, "filename": file.filename})
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    # ── Extract metadata ─────────────────────────────────────────────
    temp_path = Path(tempfile.gettempdir()) / f"{cache_key}_meta{ext}"
    temp_path.write_bytes(content)
    try:
        metadata = decoder.extract_metadata(str(temp_path), img)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    # ── Generate "before" image (flat RAW render) ────────────────────
    before_preview = resize_to_fit(img, 1200)
    before_b64 = encode_jpeg_base64(before_preview, quality=85)

    # ── Run ML pipeline ──────────────────────────────────────────────
    pipeline = _get_pipeline()
    if pipeline is None or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="ML pipeline not ready")

    def process_fn(patch):
        result, _ = pipeline.process(patch, profile)
        return result

    # Use patch processing for large images
    if patch_processor.needs_patching(img):
        t0 = time.time()
        enhanced = patch_processor.process(img, process_fn)
        dt = (time.time() - t0) * 1000
        # Get stage timings from a single patch run for reporting
        _, patch_stages = pipeline.process(
            resize_to_fit(img, 512), profile
        )
        stages.extend(patch_stages)
    else:
        enhanced, pipeline_stages = pipeline.process(img, profile)
        stages.extend(pipeline_stages)

    # ── Encode result ────────────────────────────────────────────────
    after_preview = resize_to_fit(enhanced, 1200)
    after_b64 = encode_jpeg_base64(after_preview, quality=OUTPUT_JPEG_QUALITY)

    total_time = round((time.time() - total_start) * 1000, 1)

    session_id = cache_key

    return EnhancementResponse(
        result=after_b64,
        before=before_b64,
        session_id=session_id,
        profile=profile,
        processing_time=total_time,
        stages=stages,
        metadata=metadata,
    )


@router.get("/profiles", response_model=list[ProfileInfo])
async def list_profiles():
    """Return all available enhancement profiles."""
    return PROFILE_INFO


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pipeline = _get_pipeline()
    return HealthResponse(
        status="ok",
        pipeline_mode=pipeline.mode if pipeline else "unknown",
        models_loaded=pipeline.is_loaded if pipeline else False,
    )
