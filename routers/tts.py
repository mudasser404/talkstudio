"""
TTS Router - Voice Generation API Endpoints
"""

import base64
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from models.schemas import (
    GenerateRequest,
    GenerateResponse,
    BatchGenerateRequest,
    BatchGenerateResponse,
    ModelLoadRequest,
    ModelListResponse,
    ModelInfo
)
from services.tts_service import tts_service
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tts", tags=["TTS - Voice Generation"])


@router.post("/generate", response_model=GenerateResponse)
async def generate_speech(request: GenerateRequest):
    """
    Generate speech from text using F5-TTS

    - **text**: Text to synthesize (max 50000 chars)
    - **reference_audio**: Base64 encoded reference audio (WAV/MP3)
    - **reference_text**: Optional transcript of reference audio
    - **speed**: Speech speed (0.5-2.0, default: 1.0)
    - **nfe_step**: NFE steps (8-128, default: 32)
    - **cfg_strength**: CFG strength (0-5, default: 2.0)
    - **clean_audio**: Apply noise reduction (default: true)
    - **remove_silence**: Remove silence from output (default: false)
    - **seed**: Random seed for reproducibility
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


@router.post("/generate/file", response_model=GenerateResponse)
async def generate_speech_with_file(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: str = Form(default=""),
    speed: float = Form(default=1.0),
    nfe_step: int = Form(default=32),
    cfg_strength: float = Form(default=2.0),
    clean_audio: bool = Form(default=True),
    remove_silence: bool = Form(default=False),
    seed: Optional[int] = Form(default=None)
):
    """
    Generate speech from text with file upload for reference audio

    Use this endpoint when uploading reference audio as a file instead of base64.
    """
    # Read and encode audio file
    audio_bytes = await reference_audio.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    audio_data, duration, sample_rate, file_size, used_seed, error = tts_service.generate(
        text=text,
        reference_audio_base64=audio_base64,
        reference_text=reference_text,
        speed=speed,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        clean_audio=clean_audio,
        remove_silence=remove_silence,
        seed=seed
    )

    if error:
        return GenerateResponse(success=False, error=error)

    return GenerateResponse(
        success=True,
        audio_data=audio_data,
        duration=duration,
        sample_rate=sample_rate,
        file_size=file_size,
        seed=used_seed
    )


@router.post("/generate/batch", response_model=BatchGenerateResponse)
async def generate_speech_batch(request: BatchGenerateRequest):
    """
    Generate speech for multiple texts in batch

    - **texts**: List of texts to synthesize
    - **reference_audio**: Base64 encoded reference audio (used for all texts)
    - **reference_text**: Transcript of reference audio

    Returns results for each text in the same order.
    """
    results = []
    completed = 0
    failed = 0

    for text in request.texts:
        audio_data, duration, sample_rate, file_size, seed, error = tts_service.generate(
            text=text,
            reference_audio_base64=request.reference_audio,
            reference_text=request.reference_text,
            speed=request.speed,
            nfe_step=request.nfe_step,
            cfg_strength=request.cfg_strength
        )

        if error:
            results.append(GenerateResponse(success=False, error=error))
            failed += 1
        else:
            results.append(GenerateResponse(
                success=True,
                audio_data=audio_data,
                duration=duration,
                sample_rate=sample_rate,
                file_size=file_size,
                seed=seed
            ))
            completed += 1

    return BatchGenerateResponse(
        success=failed == 0,
        results=results,
        total=len(request.texts),
        completed=completed,
        failed=failed
    )


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(default=None)
):
    """
    Transcribe audio to text using Whisper

    - **audio**: Audio file to transcribe
    - **language**: Optional language code (e.g., 'en', 'zh')
    """
    import tempfile
    import os

    temp_path = None
    try:
        # Save uploaded file
        temp_path = os.path.join(tempfile.gettempdir(), f"transcribe_{audio.filename}")
        with open(temp_path, 'wb') as f:
            f.write(await audio.read())

        # Transcribe
        transcript = tts_service.transcribe(temp_path, language)

        return {"success": True, "transcript": transcript}

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"success": False, "error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/model/load")
async def load_model(request: ModelLoadRequest):
    """
    Load a specific model checkpoint

    - **model_path**: Path to model checkpoint file
    - **model_type**: Model type (F5TTS_v1_Base, F5TTS_Base, E2TTS_Base)
    """
    # Unload current model first
    tts_service.unload_model()

    # Load new model
    success = tts_service.load_model(request.model_path, request.model_type)

    if success:
        return {"success": True, "message": "Model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")


@router.post("/model/reload")
async def reload_default_model():
    """
    Reload the default F5-TTS model

    Use this to reload the default model after installation or if model failed to load initially.
    """
    # Unload current model first
    tts_service.unload_model()

    # Load default model
    success = tts_service.load_model()

    if success:
        return {"success": True, "message": "Default model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load default model")


@router.post("/model/unload")
async def unload_model():
    """Unload the current model to free memory"""
    tts_service.unload_model()
    return {"success": True, "message": "Model unloaded"}


@router.get("/model/status")
async def get_model_status():
    """Get current model status"""
    gpu_info = tts_service.get_gpu_info()

    return {
        "model_loaded": tts_service.is_model_loaded,
        "device": tts_service.device,
        "gpu_available": gpu_info["available"],
        "gpu_name": gpu_info.get("name"),
        "gpu_memory_total": gpu_info.get("memory_total"),
        "gpu_memory_used": gpu_info.get("memory_used")
    }