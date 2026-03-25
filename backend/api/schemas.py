"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ProfileName(str, Enum):
    EXPERT_NATURAL = "expert_natural"
    WARM_FILM = "warm_film"
    MOODY_CONTRAST = "moody_contrast"
    BW_FINE_ART = "bw_fine_art"
    GOLDEN_HOUR = "golden_hour"
    CLEAN_COMMERCIAL = "clean_commercial"


class RawMetadata(BaseModel):
    camera_model: Optional[str] = None
    iso: Optional[int] = None
    aperture: Optional[str] = None
    shutter_speed: Optional[str] = None
    focal_length: Optional[str] = None
    width: int = 0
    height: int = 0
    file_format: str = ""
    file_size_mb: float = 0.0


class StageTime(BaseModel):
    name: str
    duration_ms: float


class EnhancementResponse(BaseModel):
    result: str  # base64 encoded JPEG
    before: str  # base64 encoded before image (flat RAW render)
    session_id: str
    profile: str
    processing_time: float
    stages: List[StageTime]
    metadata: RawMetadata


class ProfileInfo(BaseModel):
    id: str
    name: str
    description: str
    aesthetic: str


class HealthResponse(BaseModel):
    status: str
    pipeline_mode: str
    models_loaded: bool
