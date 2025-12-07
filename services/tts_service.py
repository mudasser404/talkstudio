"""
TTS Service - Voice Generation Logic
"""

import os
import sys
import uuid
import base64
import tempfile
import threading
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

# Add F5-TTS to path
F5_TTS_PATH = Path(__file__).parent.parent / "F5-TTS" / "src"
if str(F5_TTS_PATH) not in sys.path:
    sys.path.insert(0, str(F5_TTS_PATH))

from config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Service for TTS generation using F5-TTS"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._model_lock = threading.Lock()
        self._device = None
        self._initialized = True
        logger.info("TTSService initialized")

    @property
    def device(self) -> str:
        """Get the device being used"""
        if self._device is None:
            import torch
            if settings.DEVICE:
                self._device = settings.DEVICE
            elif torch.cuda.is_available():
                self._device = "cuda:0"
            else:
                self._device = "cpu"
        return self._device

    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None

    def load_model(self, model_path: Optional[str] = None, model_type: str = "F5TTS_v1_Base") -> bool:
        """Load F5-TTS model"""
        with self._model_lock:
            if self._model is not None:
                logger.info("Model already loaded")
                return True

            try:
                import torch
                from f5_tts.api import F5TTS

                logger.info(f"Loading F5-TTS model on {self.device}...")

                # Load model
                if model_path:
                    self._model = F5TTS(
                        model=model_type,
                        ckpt_file=model_path,
                        device=self.device
                    )
                else:
                    self._model = F5TTS(
                        model=model_type,
                        device=self.device
                    )

                logger.info("F5-TTS model loaded successfully!")
                return True

            except Exception as e:
                logger.error(f"Failed to load F5-TTS model: {e}", exc_info=True)
                return False

    def unload_model(self):
        """Unload the model to free memory"""
        with self._model_lock:
            if self._model is not None:
                import torch
                del self._model
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Model unloaded")

    def get_gpu_info(self) -> dict:
        """Get GPU information"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "available": True,
                    "name": torch.cuda.get_device_name(0),
                    "memory_total": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
                    "memory_used": torch.cuda.memory_allocated(0) // (1024 * 1024)
                }
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
        return {"available": False, "name": None, "memory_total": None, "memory_used": None}

    def preprocess_audio(self, audio_path: str, target_sr: int = 24000) -> str:
        """Preprocess audio: convert to proper format"""
        try:
            audio_data, sr = sf.read(audio_path)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample if needed
            if sr != target_sr:
                try:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                except ImportError:
                    logger.warning("librosa not installed, skipping resample")

            # Normalize audio
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95

            # Save preprocessed audio
            processed_path = audio_path.replace('.wav', '_processed.wav')
            sf.write(processed_path, audio_data, target_sr)

            return processed_path

        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}, using original")
            return audio_path

    def apply_noise_reduction(self, audio_path: str, strength: float = 0.3) -> str:
        """Apply noise reduction to audio"""
        try:
            import noisereduce as nr

            audio_data, sr = sf.read(audio_path)

            # Apply noise reduction
            reduced = nr.reduce_noise(
                y=audio_data,
                sr=sr,
                prop_decrease=strength,
                stationary=True
            )

            # Save cleaned audio
            cleaned_path = audio_path.replace('.wav', '_cleaned.wav')
            sf.write(cleaned_path, reduced, sr)

            return cleaned_path

        except ImportError:
            logger.warning("noisereduce not installed, skipping noise reduction")
            return audio_path
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio_path

    def generate(
        self,
        text: str,
        reference_audio_base64: str,
        reference_text: str = "",
        speed: float = 1.0,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        clean_audio: bool = True,
        noise_reduction_strength: float = 0.3,
        remove_silence: bool = False,
        seed: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[float], Optional[int], Optional[int], Optional[int], Optional[str]]:
        """
        Generate speech from text

        Returns: (audio_base64, duration, sample_rate, file_size, seed, error)
        """
        temp_files = []

        try:
            # Ensure model is loaded
            if not self.is_model_loaded:
                if not self.load_model():
                    return None, None, None, None, None, "Failed to load model"

            # Validate input
            if not text.strip():
                return None, None, None, None, None, "Text cannot be empty"

            if len(text) > settings.MAX_TEXT_LENGTH:
                return None, None, None, None, None, f"Text too long (max {settings.MAX_TEXT_LENGTH} chars)"

            # Decode reference audio
            try:
                audio_bytes = base64.b64decode(reference_audio_base64)
            except Exception as e:
                return None, None, None, None, None, f"Invalid base64 audio: {e}"

            if len(audio_bytes) < 1000:
                return None, None, None, None, None, "Reference audio too small"

            # Save reference audio to temp file
            ref_audio_path = os.path.join(tempfile.gettempdir(), f"ref_{uuid.uuid4().hex}.wav")
            temp_files.append(ref_audio_path)

            with open(ref_audio_path, 'wb') as f:
                f.write(audio_bytes)

            # Preprocess audio
            processed_audio = self.preprocess_audio(ref_audio_path)
            if processed_audio != ref_audio_path:
                temp_files.append(processed_audio)

            # Apply noise reduction if requested
            if clean_audio:
                cleaned_audio = self.apply_noise_reduction(processed_audio, noise_reduction_strength)
                if cleaned_audio != processed_audio:
                    temp_files.append(cleaned_audio)
                processed_audio = cleaned_audio

            # Generate speech
            logger.info(f"Generating speech: text_len={len(text)}, speed={speed}")

            output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4().hex}.wav")
            temp_files.append(output_path)

            with self._model_lock:
                wav, sr, _ = self._model.infer(
                    ref_file=processed_audio,
                    ref_text=reference_text,
                    gen_text=text,
                    nfe_step=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    speed=speed,
                    remove_silence=remove_silence,
                    seed=seed
                )
                used_seed = self._model.seed

            # Save output
            sf.write(output_path, wav, sr)

            # Read and encode output
            with open(output_path, 'rb') as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')

            # Get file info
            file_size = os.path.getsize(output_path)
            duration = len(wav) / sr

            logger.info(f"Generation complete: duration={duration:.2f}s")

            return audio_data, duration, sr, file_size, used_seed, None

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return None, None, None, None, None, str(e)

        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio to text"""
        if not self.is_model_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load model")

        with self._model_lock:
            return self._model.transcribe(audio_path, language)


# Global service instance
tts_service = TTSService()