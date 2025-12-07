"""
RunPod Serverless Handler for F5-TTS Voice Cloning API
Deploy this on RunPod Serverless with GPU
"""

import os
import sys
import uuid
import base64
import tempfile
import logging
import requests
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

F5_TTS_PATH = PROJECT_ROOT / "F5-TTS" / "src"
if str(F5_TTS_PATH) not in sys.path:
    sys.path.insert(0, str(F5_TTS_PATH))

# Global model instance
model = None


def load_model():
    """Load F5-TTS model (called once during cold start)"""
    global model
    if model is not None:
        return model

    try:
        import torch
        from f5_tts.api import F5TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading F5-TTS model on {device}...")

        model = F5TTS(
            model="F5TTS_v1_Base",
            device=device
        )

        logger.info("F5-TTS model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def download_audio_from_url(url: str) -> bytes:
    """Download audio file from URL"""
    try:
        logger.info(f"Downloading audio from: {url}")
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('content-type', '')
        logger.info(f"Content-Type: {content_type}")

        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download audio: {e}")
        raise Exception(f"Failed to download audio from URL: {str(e)}")


def generate_speech(
    job_id: str,
    text: str,
    reference_audio_url: str,
    reference_text: str = "",
    speed: float = 1.0,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
    seed: int = None
):
    """Generate speech using F5-TTS"""
    import torch
    import soundfile as sf

    global model
    if model is None:
        model = load_model()

    temp_ref_path = None
    temp_out_path = None
    started_at = datetime.utcnow().isoformat()

    try:
        # Download reference audio from URL
        audio_bytes = download_audio_from_url(reference_audio_url)

        # Determine file extension from URL or default to wav
        url_lower = reference_audio_url.lower()
        if '.mp3' in url_lower:
            ext = '.mp3'
        elif '.wav' in url_lower:
            ext = '.wav'
        elif '.flac' in url_lower:
            ext = '.flac'
        elif '.ogg' in url_lower:
            ext = '.ogg'
        else:
            ext = '.wav'

        # Save to temp file
        temp_ref_path = tempfile.mktemp(suffix=ext)
        with open(temp_ref_path, 'wb') as f:
            f.write(audio_bytes)

        logger.info(f"Reference audio saved to: {temp_ref_path}, size: {len(audio_bytes)} bytes")

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        # Generate audio
        temp_out_path = tempfile.mktemp(suffix=".wav")

        logger.info(f"Generating speech for text: {text[:50]}...")

        wav, sr, _ = model.infer(
            ref_file=temp_ref_path,
            ref_text=reference_text,
            gen_text=text,
            file_wave=temp_out_path,
            speed=speed,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef
        )

        # Read generated audio
        audio_data, sample_rate = sf.read(temp_out_path)

        # Calculate duration
        duration = len(audio_data) / sample_rate

        # Encode to base64
        with open(temp_out_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        file_size = os.path.getsize(temp_out_path)
        completed_at = datetime.utcnow().isoformat()

        logger.info(f"Speech generated successfully! Duration: {duration:.2f}s")

        return {
            "success": True,
            "job_id": job_id,
            "status": "completed",
            "audio_base64": audio_base64,
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "file_size": file_size,
            "seed": seed,
            "started_at": started_at,
            "completed_at": completed_at,
            "parameters": {
                "text": text,
                "reference_audio_url": reference_audio_url,
                "reference_text": reference_text,
                "speed": speed,
                "nfe_step": nfe_step,
                "cfg_strength": cfg_strength,
                "sway_sampling_coef": sway_sampling_coef
            }
        }

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {
            "success": False,
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "started_at": started_at,
            "completed_at": datetime.utcnow().isoformat()
        }

    finally:
        # Cleanup temp files
        if temp_ref_path and os.path.exists(temp_ref_path):
            os.remove(temp_ref_path)
        if temp_out_path and os.path.exists(temp_out_path):
            os.remove(temp_out_path)


def handler(job):
    """
    RunPod Serverless Handler

    Expected input format:
    {
        "input": {
            "text": "Text to synthesize",
            "reference_audio_url": "https://example.com/audio.wav",
            "reference_text": "Optional transcript of reference audio",
            "speed": 1.0,
            "nfe_step": 32,
            "cfg_strength": 2.0,
            "sway_sampling_coef": -1.0,
            "seed": null
        }
    }

    Response format:
    {
        "success": true,
        "job_id": "unique-job-id",
        "status": "completed",
        "audio_base64": "base64_encoded_wav_audio",
        "duration": 3.5,
        "sample_rate": 24000,
        "file_size": 168000,
        "seed": 12345678,
        "started_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T00:00:05",
        "parameters": {
            "text": "...",
            "reference_audio_url": "...",
            ...
        }
    }
    """
    job_input = job.get("input", {})

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Check for status action
    action = job_input.get("action", "generate")

    if action == "status":
        import torch
        return {
            "success": True,
            "job_id": job_id,
            "status": "ready",
            "model_loaded": model is not None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    # Validate required parameters
    text = job_input.get("text")
    reference_audio_url = job_input.get("reference_audio_url")

    if not text:
        return {
            "success": False,
            "job_id": job_id,
            "status": "failed",
            "error": "Missing required parameter: 'text'"
        }

    if not reference_audio_url:
        return {
            "success": False,
            "job_id": job_id,
            "status": "failed",
            "error": "Missing required parameter: 'reference_audio_url'"
        }

    # Generate speech
    return generate_speech(
        job_id=job_id,
        text=text,
        reference_audio_url=reference_audio_url,
        reference_text=job_input.get("reference_text", ""),
        speed=job_input.get("speed", 1.0),
        nfe_step=job_input.get("nfe_step", 32),
        cfg_strength=job_input.get("cfg_strength", 2.0),
        sway_sampling_coef=job_input.get("sway_sampling_coef", -1.0),
        seed=job_input.get("seed")
    )


# RunPod entry point
if __name__ == "__main__":
    import runpod

    # Pre-load model during cold start
    logger.info("Pre-loading model...")
    load_model()
    logger.info("Model ready, starting RunPod handler...")

    runpod.serverless.start({"handler": handler})