"""
Dataset Service - Dataset Preparation Logic
"""

import os
import sys
import uuid
import json
import shutil
import threading
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import soundfile as sf

# Add F5-TTS to path
F5_TTS_PATH = Path(__file__).parent.parent / "F5-TTS" / "src"
if str(F5_TTS_PATH) not in sys.path:
    sys.path.insert(0, str(F5_TTS_PATH))

from config import settings

logger = logging.getLogger(__name__)


class DatasetStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DatasetInfo:
    """Dataset information"""
    dataset_id: str
    name: str
    description: str
    language: str
    status: DatasetStatus
    total_samples: int = 0
    total_duration: float = 0.0  # in hours
    raw_path: Optional[str] = None
    prepared_path: Optional[str] = None
    is_finetune: bool = True
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data['status'] = self.status.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data


class DatasetService:
    """Service for dataset management"""

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

        self._datasets: Dict[str, DatasetInfo] = {}
        self._datasets_lock = threading.Lock()
        self._datasets_file = settings.DATASETS_DIR / "datasets.json"
        self._initialized = True

        # Load existing datasets from file
        self._load_datasets()
        logger.info("DatasetService initialized")

    def _load_datasets(self):
        """Load datasets from persistent storage"""
        if self._datasets_file.exists():
            try:
                with open(self._datasets_file, 'r') as f:
                    data = json.load(f)
                for ds_data in data.get('datasets', []):
                    ds_data['status'] = DatasetStatus(ds_data['status'])
                    if ds_data.get('created_at'):
                        ds_data['created_at'] = datetime.fromisoformat(ds_data['created_at'])
                    if ds_data.get('updated_at'):
                        ds_data['updated_at'] = datetime.fromisoformat(ds_data['updated_at'])
                    ds = DatasetInfo(**ds_data)
                    self._datasets[ds.dataset_id] = ds
            except Exception as e:
                logger.error(f"Failed to load datasets: {e}")

    def _save_datasets(self):
        """Save datasets to persistent storage"""
        try:
            data = {
                'datasets': [ds.to_dict() for ds in self._datasets.values()]
            }
            with open(self._datasets_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save datasets: {e}")

    def create_dataset(
        self,
        name: str,
        description: str = "",
        language: str = "en",
        is_finetune: bool = True
    ) -> DatasetInfo:
        """Create a new dataset"""
        dataset_id = str(uuid.uuid4())[:8]

        # Create dataset directories
        raw_path = settings.DATASETS_DIR / f"{dataset_id}_raw"
        raw_path.mkdir(parents=True, exist_ok=True)
        (raw_path / "wavs").mkdir(exist_ok=True)

        dataset = DatasetInfo(
            dataset_id=dataset_id,
            name=name,
            description=description,
            language=language,
            status=DatasetStatus.PENDING,
            raw_path=str(raw_path),
            is_finetune=is_finetune,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        with self._datasets_lock:
            self._datasets[dataset_id] = dataset
            self._save_datasets()

        logger.info(f"Created dataset {dataset_id}: {name}")
        return dataset

    def add_audio(
        self,
        dataset_id: str,
        audio_data: bytes,
        filename: str,
        text: str
    ) -> Tuple[bool, Optional[str]]:
        """Add an audio file to the dataset"""
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            return False, "Dataset not found"

        if dataset.status not in [DatasetStatus.PENDING, DatasetStatus.UPLOADING]:
            return False, "Dataset is not in upload state"

        try:
            # Update status
            dataset.status = DatasetStatus.UPLOADING
            dataset.updated_at = datetime.now()

            # Save audio file
            wavs_dir = Path(dataset.raw_path) / "wavs"
            audio_path = wavs_dir / filename

            with open(audio_path, 'wb') as f:
                f.write(audio_data)

            # Get audio duration
            try:
                info = sf.info(str(audio_path))
                duration = info.duration
            except Exception:
                duration = 0

            # Update or create metadata.csv
            metadata_path = Path(dataset.raw_path) / "metadata.csv"
            file_exists = metadata_path.exists()

            with open(metadata_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='|')
                if not file_exists:
                    writer.writerow(['audio_file', 'text'])
                writer.writerow([f"wavs/{filename}", text])

            # Update dataset stats
            dataset.total_samples += 1
            dataset.total_duration += duration / 3600  # convert to hours
            dataset.updated_at = datetime.now()

            with self._datasets_lock:
                self._save_datasets()

            return True, None

        except Exception as e:
            logger.error(f"Failed to add audio to dataset: {e}")
            return False, str(e)

    def add_audio_batch(
        self,
        dataset_id: str,
        audio_files: List[Tuple[bytes, str, str]]  # (audio_data, filename, text)
    ) -> Tuple[int, int, List[str]]:
        """Add multiple audio files to dataset

        Returns: (success_count, fail_count, errors)
        """
        success = 0
        failed = 0
        errors = []

        for audio_data, filename, text in audio_files:
            ok, error = self.add_audio(dataset_id, audio_data, filename, text)
            if ok:
                success += 1
            else:
                failed += 1
                errors.append(f"{filename}: {error}")

        return success, failed, errors

    def prepare_dataset(self, dataset_id: str) -> Tuple[bool, Optional[str]]:
        """Prepare dataset for training (converts to arrow format)"""
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            return False, "Dataset not found"

        if dataset.status == DatasetStatus.PROCESSING:
            return False, "Dataset is already being processed"

        if dataset.total_samples == 0:
            return False, "Dataset has no samples"

        # Start preparation in background
        thread = threading.Thread(target=self._prepare_dataset_async, args=(dataset_id,))
        thread.daemon = True
        thread.start()

        return True, None

    def _prepare_dataset_async(self, dataset_id: str):
        """Prepare dataset asynchronously"""
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            return

        try:
            dataset.status = DatasetStatus.PROCESSING
            dataset.updated_at = datetime.now()
            self._save_datasets()

            # Import preparation function
            from f5_tts.train.datasets.prepare_csv_wavs import prepare_and_save_set

            # Create output directory
            prepared_path = settings.DATASETS_DIR / f"{dataset_id}_prepared"
            prepared_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Preparing dataset {dataset_id}...")

            # Run preparation
            prepare_and_save_set(
                inp_dir=dataset.raw_path,
                out_dir=str(prepared_path),
                is_finetune=dataset.is_finetune
            )

            dataset.prepared_path = str(prepared_path)
            dataset.status = DatasetStatus.COMPLETED
            logger.info(f"Dataset {dataset_id} prepared successfully")

        except Exception as e:
            dataset.status = DatasetStatus.FAILED
            dataset.error = str(e)
            logger.error(f"Failed to prepare dataset {dataset_id}: {e}", exc_info=True)

        finally:
            dataset.updated_at = datetime.now()
            with self._datasets_lock:
                self._save_datasets()

    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get dataset by ID"""
        return self._datasets.get(dataset_id)

    def list_datasets(self) -> List[DatasetInfo]:
        """List all datasets"""
        return list(self._datasets.values())

    def delete_dataset(self, dataset_id: str, delete_files: bool = True) -> bool:
        """Delete a dataset"""
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            return False

        if dataset.status == DatasetStatus.PROCESSING:
            return False

        # Delete files if requested
        if delete_files:
            if dataset.raw_path and os.path.exists(dataset.raw_path):
                try:
                    shutil.rmtree(dataset.raw_path)
                except Exception as e:
                    logger.error(f"Failed to delete raw dataset files: {e}")

            if dataset.prepared_path and os.path.exists(dataset.prepared_path):
                try:
                    shutil.rmtree(dataset.prepared_path)
                except Exception as e:
                    logger.error(f"Failed to delete prepared dataset files: {e}")

        with self._datasets_lock:
            del self._datasets[dataset_id]
            self._save_datasets()

        return True

    def get_dataset_samples(self, dataset_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get samples from dataset"""
        dataset = self._datasets.get(dataset_id)
        if not dataset or not dataset.raw_path:
            return []

        samples = []
        metadata_path = Path(dataset.raw_path) / "metadata.csv"

        if not metadata_path.exists():
            return []

        try:
            with open(metadata_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f, delimiter='|')
                next(reader)  # Skip header
                for i, row in enumerate(reader):
                    if i < offset:
                        continue
                    if len(samples) >= limit:
                        break
                    if len(row) >= 2:
                        audio_file = row[0].strip()
                        text = row[1].strip()

                        # Get duration if possible
                        audio_path = Path(dataset.raw_path) / audio_file
                        duration = 0
                        if audio_path.exists():
                            try:
                                info = sf.info(str(audio_path))
                                duration = info.duration
                            except:
                                pass

                        samples.append({
                            "index": i,
                            "audio_file": audio_file,
                            "text": text,
                            "duration": duration
                        })
        except Exception as e:
            logger.error(f"Failed to read dataset samples: {e}")

        return samples

    def import_from_folder(
        self,
        dataset_id: str,
        folder_path: str,
        metadata_file: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Import dataset from existing folder

        The folder should contain:
        - wavs/ directory with audio files
        - metadata.csv file with format: audio_file|text
        """
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            return False, "Dataset not found"

        folder = Path(folder_path)
        if not folder.exists():
            return False, "Folder not found"

        try:
            # Find metadata file
            if metadata_file:
                meta_path = folder / metadata_file
            else:
                meta_path = folder / "metadata.csv"

            if not meta_path.exists():
                return False, "metadata.csv not found in folder"

            # Find wavs directory
            wavs_dir = folder / "wavs"
            if not wavs_dir.exists():
                wavs_dir = folder  # Try root folder

            # Copy files to dataset
            dest_wavs = Path(dataset.raw_path) / "wavs"
            dest_meta = Path(dataset.raw_path) / "metadata.csv"

            # Copy metadata
            shutil.copy2(meta_path, dest_meta)

            # Copy audio files and count
            total_duration = 0
            count = 0

            with open(meta_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f, delimiter='|')
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        audio_file = row[0].strip()
                        src_audio = folder / audio_file

                        if src_audio.exists():
                            dest_audio = dest_wavs / Path(audio_file).name
                            shutil.copy2(src_audio, dest_audio)

                            try:
                                info = sf.info(str(dest_audio))
                                total_duration += info.duration
                            except:
                                pass

                            count += 1

            dataset.total_samples = count
            dataset.total_duration = total_duration / 3600
            dataset.status = DatasetStatus.PENDING
            dataset.updated_at = datetime.now()

            with self._datasets_lock:
                self._save_datasets()

            return True, None

        except Exception as e:
            logger.error(f"Failed to import dataset: {e}")
            return False, str(e)


# Global service instance
dataset_service = DatasetService()