"""
FastAPI application entry point for the RAW Photo Enhancement Pipeline.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import CORS_ORIGINS, OUTPUT_DIR, PIPELINE_MODE
from backend.api.routes import router
from backend.models.ml_pipeline import MLPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: MLPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML pipeline on startup, cleanup on shutdown."""
    global pipeline
    logger.info(f"Initializing ML pipeline (mode: {PIPELINE_MODE})")
    pipeline = MLPipeline(mode=PIPELINE_MODE)
    pipeline.load()
    logger.info("Pipeline ready")
    yield
    logger.info("Shutting down, cleaning up cache...")
    pipeline = None


app = FastAPI(
    title="AI RAW Photo Enhancement",
    description="Three-stage ML pipeline: Denoise → Sharpen → Color Grade",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Serve output images
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


def get_pipeline() -> MLPipeline:
    return pipeline
