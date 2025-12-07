"""
F5-TTS API Server
Standalone FastAPI server for voice cloning and model training using F5-TTS model
Deploy this on a GPU server
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add F5-TTS to path
F5_TTS_PATH = PROJECT_ROOT / "F5-TTS" / "src"
if str(F5_TTS_PATH) not in sys.path:
    sys.path.insert(0, str(F5_TTS_PATH))

from config import settings
from routers import tts_router, training_router, dataset_router
from services.tts_service import tts_service
from services.training_service import training_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOGS_DIR / "server.log")
    ]
)
logger = logging.getLogger(__name__)

# Server start time
START_TIME = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("F5-TTS API Server starting...")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"F5-TTS path: {F5_TTS_PATH}")

    # Preload model in background
    import threading
    def load_model_background():
        try:
            tts_service.load_model()
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")

    thread = threading.Thread(target=load_model_background)
    thread.start()
    logger.info("Model loading initiated in background...")

    yield

    # Shutdown
    logger.info("F5-TTS API Server shutting down...")
    tts_service.unload_model()


# Initialize FastAPI app
app = FastAPI(
    title="F5-TTS Voice Cloning API",
    description="""
    ## F5-TTS API Server

    A comprehensive API server for voice cloning and model training using F5-TTS.

    ### Features:
    - **Voice Generation**: Generate speech from text using reference audio
    - **Batch Processing**: Generate multiple audio files in one request
    - **Dataset Management**: Create, upload, and prepare training datasets
    - **Model Training**: Fine-tune F5-TTS model on custom datasets
    - **Checkpoint Management**: Track and manage training checkpoints

    ### Quick Start:
    1. Create a dataset: `POST /api/datasets/create`
    2. Upload audio files: `POST /api/datasets/{id}/upload`
    3. Prepare dataset: `POST /api/datasets/{id}/prepare`
    4. Start training: `POST /api/training/start`
    5. Monitor progress: `GET /api/training/jobs/{id}`
    6. Generate speech: `POST /api/tts/generate`
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Include routers
app.include_router(tts_router)
app.include_router(training_router)
app.include_router(dataset_router)


# ==================== UI Endpoint ====================

@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    """Serve the web UI"""
    ui_path = PROJECT_ROOT / "templates" / "index.html"
    if ui_path.exists():
        return ui_path.read_text(encoding='utf-8')
    return HTMLResponse("<h1>UI not found</h1><p>Please check templates/index.html</p>", status_code=404)


# ==================== Root Endpoints ====================

@app.get("/", tags=["Status"])
async def root():
    """Root endpoint - returns server info"""
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "name": "F5-TTS API Server",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": tts_service.is_model_loaded,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "active_training_jobs": training_service.get_active_jobs_count(),
        "ui_url": "/ui",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.get("/status", tags=["Status"])
async def get_status():
    """Get detailed server status"""
    gpu_info = tts_service.get_gpu_info()
    uptime = (datetime.now() - START_TIME).total_seconds()

    return {
        "status": "running",
        "uptime_seconds": uptime,
        "model_loaded": tts_service.is_model_loaded,
        "device": tts_service.device,
        "gpu": gpu_info,
        "active_training_jobs": training_service.get_active_jobs_count(),
        "version": "2.0.0"
    }


@app.get("/health", tags=["Status"])
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ==================== Legacy Endpoints (Backward Compatibility) ====================

from models.schemas import GenerateRequest, GenerateResponse

@app.post("/generate", response_model=GenerateResponse, tags=["Legacy"])
async def generate_speech_legacy(request: GenerateRequest):
    """
    Legacy endpoint for voice generation (use /api/tts/generate instead)
    """
    audio_data, duration, sample_rate, file_size, seed, error = tts_service.generate(
        text=request.text,
        reference_audio_base64=request.reference_audio,
        reference_text=request.reference_text,
        speed=request.speed,
        nfe_step=request.nfe_step,
        cfg_strength=request.cfg_strength,
        sway_sampling_coef=request.sway_sampling_coef,
        clean_audio=request.clean_audio,
        noise_reduction_strength=request.noise_reduction_strength,
        remove_silence=request.remove_silence,
        seed=request.seed
    )

    if error:
        return GenerateResponse(success=False, error=error)

    return GenerateResponse(
        success=True,
        audio_data=audio_data,
        duration=duration,
        sample_rate=sample_rate,
        file_size=file_size,
        seed=seed
    )


# ==================== Main ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F5-TTS API Server")
    parser.add_argument("--host", default=settings.HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind")
    parser.add_argument("--workers", type=int, default=settings.WORKERS, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              F5-TTS Voice Cloning API Server v2.0                ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Host: {args.host:<15}  Port: {args.port:<10}                    ║
    ║                                                                  ║
    ║  Web UI:       http://{args.host}:{args.port}/ui                     ║
    ║  API Docs:     http://{args.host}:{args.port}/docs                   ║
    ║  ReDoc:        http://{args.host}:{args.port}/redoc                  ║
    ║                                                                  ║
    ║  Endpoints:                                                      ║
    ║    - TTS Generation:  /api/tts/*                                 ║
    ║    - Training:        /api/training/*                            ║
    ║    - Datasets:        /api/datasets/*                            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )