"""
Training Service - Model Training Logic
"""

import os
import sys
import uuid
import json
import shutil
import threading
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Add F5-TTS to path
F5_TTS_PATH = Path(__file__).parent.parent / "F5-TTS" / "src"
if str(F5_TTS_PATH) not in sys.path:
    sys.path.insert(0, str(F5_TTS_PATH))

from config import settings

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job information"""
    job_id: str
    dataset_id: str
    dataset_path: str
    status: TrainingStatus
    config: Dict[str, Any]
    current_epoch: int = 0
    current_update: int = 0
    total_updates: int = 0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    process_pid: Optional[int] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data['status'] = self.status.value
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data


class TrainingService:
    """Service for model training"""

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

        self._jobs: Dict[str, TrainingJob] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._jobs_lock = threading.Lock()
        self._jobs_file = settings.CHECKPOINTS_DIR / "training_jobs.json"
        self._initialized = True

        # Load existing jobs from file
        self._load_jobs()
        logger.info("TrainingService initialized")

    def _load_jobs(self):
        """Load jobs from persistent storage"""
        if self._jobs_file.exists():
            try:
                with open(self._jobs_file, 'r') as f:
                    data = json.load(f)
                for job_data in data.get('jobs', []):
                    job_data['status'] = TrainingStatus(job_data['status'])
                    if job_data.get('started_at'):
                        job_data['started_at'] = datetime.fromisoformat(job_data['started_at'])
                    if job_data.get('updated_at'):
                        job_data['updated_at'] = datetime.fromisoformat(job_data['updated_at'])
                    job = TrainingJob(**job_data)
                    # Mark running jobs as failed if server restarted
                    if job.status == TrainingStatus.TRAINING:
                        job.status = TrainingStatus.FAILED
                        job.error = "Server restarted during training"
                    self._jobs[job.job_id] = job
            except Exception as e:
                logger.error(f"Failed to load training jobs: {e}")

    def _save_jobs(self):
        """Save jobs to persistent storage"""
        try:
            data = {
                'jobs': [job.to_dict() for job in self._jobs.values()]
            }
            with open(self._jobs_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training jobs: {e}")

    def start_training(
        self,
        dataset_id: str,
        dataset_path: str,
        config: Dict[str, Any]
    ) -> TrainingJob:
        """Start a new training job"""
        job_id = str(uuid.uuid4())[:8]

        # Create checkpoint directory for this job
        checkpoint_path = settings.CHECKPOINTS_DIR / f"job_{job_id}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        job = TrainingJob(
            job_id=job_id,
            dataset_id=dataset_id,
            dataset_path=dataset_path,
            status=TrainingStatus.PENDING,
            config=config,
            checkpoint_path=str(checkpoint_path),
            started_at=datetime.now(),
            updated_at=datetime.now()
        )

        with self._jobs_lock:
            self._jobs[job_id] = job
            self._save_jobs()

        # Start training in background thread
        thread = threading.Thread(target=self._run_training, args=(job_id,))
        thread.daemon = True
        thread.start()

        return job

    def _run_training(self, job_id: str):
        """Run training process"""
        job = self._jobs.get(job_id)
        if not job:
            return

        try:
            job.status = TrainingStatus.PREPARING
            job.updated_at = datetime.now()
            self._save_jobs()

            # Prepare training command
            config = job.config
            cmd = [
                sys.executable, "-m", "f5_tts.train.finetune_cli",
                "--exp_name", config.get("exp_name", "F5TTS_v1_Base"),
                "--dataset_name", job.dataset_path,
                "--learning_rate", str(config.get("learning_rate", 1e-5)),
                "--batch_size_per_gpu", str(config.get("batch_size_per_gpu", 3200)),
                "--batch_size_type", config.get("batch_size_type", "frame"),
                "--max_samples", str(config.get("max_samples", 64)),
                "--epochs", str(config.get("epochs", 100)),
                "--num_warmup_updates", str(config.get("num_warmup_updates", 20000)),
                "--save_per_updates", str(config.get("save_per_updates", 5000)),
                "--keep_last_n_checkpoints", str(config.get("keep_last_n_checkpoints", 5)),
                "--grad_accumulation_steps", str(config.get("grad_accumulation_steps", 1)),
                "--max_grad_norm", str(config.get("max_grad_norm", 1.0)),
                "--tokenizer", config.get("tokenizer", "pinyin"),
                "--finetune"
            ]

            if config.get("pretrained_path"):
                cmd.extend(["--pretrain", config["pretrained_path"]])

            if config.get("use_bnb_optimizer"):
                cmd.append("--bnb_optimizer")

            if config.get("logger"):
                cmd.extend(["--logger", config["logger"]])

            logger.info(f"Starting training job {job_id}: {' '.join(cmd)}")

            # Set environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(F5_TTS_PATH) + os.pathsep + env.get("PYTHONPATH", "")

            # Start training process
            job.status = TrainingStatus.TRAINING
            job.updated_at = datetime.now()
            self._save_jobs()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(F5_TTS_PATH.parent)
            )

            job.process_pid = process.pid
            self._processes[job_id] = process

            # Monitor training output
            log_file = Path(job.checkpoint_path) / "training.log"
            with open(log_file, 'w') as f:
                for line in process.stdout:
                    f.write(line)
                    f.flush()

                    # Parse training progress from output
                    self._parse_training_output(job, line)

            return_code = process.wait()

            if return_code == 0:
                job.status = TrainingStatus.COMPLETED
                logger.info(f"Training job {job_id} completed successfully")
            else:
                job.status = TrainingStatus.FAILED
                job.error = f"Training process exited with code {return_code}"
                logger.error(f"Training job {job_id} failed: {job.error}")

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            logger.error(f"Training job {job_id} failed: {e}", exc_info=True)

        finally:
            job.updated_at = datetime.now()
            if job_id in self._processes:
                del self._processes[job_id]
            with self._jobs_lock:
                self._save_jobs()

    def _parse_training_output(self, job: TrainingJob, line: str):
        """Parse training output to update job progress"""
        try:
            # Parse epoch/update from tqdm output: "Epoch 1/100: 50%|... update=1000, loss=0.123"
            if "update=" in line and "loss=" in line:
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if part.startswith("update="):
                        job.current_update = int(part.split("=")[1])
                    elif part.startswith("loss="):
                        job.loss = float(part.split("=")[1])

            # Parse epoch info
            if "Epoch " in line and "/" in line:
                try:
                    epoch_part = line.split("Epoch ")[1].split("/")[0]
                    job.current_epoch = int(epoch_part)
                except:
                    pass

            job.updated_at = datetime.now()

        except Exception as e:
            pass  # Ignore parsing errors

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job by ID"""
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[TrainingJob]:
        """List all training jobs"""
        return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status not in [TrainingStatus.PENDING, TrainingStatus.PREPARING, TrainingStatus.TRAINING]:
            return False

        # Kill the process if running
        process = self._processes.get(job_id)
        if process:
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.error(f"Failed to kill training process: {e}")

        job.status = TrainingStatus.CANCELLED
        job.updated_at = datetime.now()

        with self._jobs_lock:
            self._save_jobs()

        return True

    def delete_job(self, job_id: str, delete_checkpoints: bool = False) -> bool:
        """Delete a training job"""
        job = self._jobs.get(job_id)
        if not job:
            return False

        # Cancel if running
        if job.status == TrainingStatus.TRAINING:
            self.cancel_job(job_id)

        # Delete checkpoints if requested
        if delete_checkpoints and job.checkpoint_path:
            try:
                shutil.rmtree(job.checkpoint_path)
            except Exception as e:
                logger.error(f"Failed to delete checkpoints: {e}")

        with self._jobs_lock:
            del self._jobs[job_id]
            self._save_jobs()

        return True

    def get_checkpoints(self, job_id: str) -> List[Dict]:
        """Get list of checkpoints for a job"""
        job = self._jobs.get(job_id)
        if not job or not job.checkpoint_path:
            return []

        checkpoints = []
        checkpoint_dir = Path(job.checkpoint_path)

        if not checkpoint_dir.exists():
            return []

        for f in checkpoint_dir.iterdir():
            if f.suffix in ['.pt', '.safetensors']:
                try:
                    # Extract update number from filename
                    update = 0
                    if 'model_' in f.stem:
                        parts = f.stem.split('_')
                        for part in parts:
                            if part.isdigit():
                                update = int(part)
                                break

                    checkpoints.append({
                        "name": f.name,
                        "path": str(f),
                        "update": update,
                        "size": f.stat().st_size,
                        "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.error(f"Failed to get checkpoint info: {e}")

        return sorted(checkpoints, key=lambda x: x['update'], reverse=True)

    def get_active_jobs_count(self) -> int:
        """Get count of active training jobs"""
        return sum(1 for job in self._jobs.values()
                   if job.status in [TrainingStatus.PENDING, TrainingStatus.PREPARING, TrainingStatus.TRAINING])


# Global service instance
training_service = TrainingService()