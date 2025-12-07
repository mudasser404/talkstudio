"""
Training Router - Model Training API Endpoints
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

from models.schemas import (
    TrainingConfig,
    TrainingStartRequest,
    TrainingResponse,
    TrainingListResponse,
    TrainingStatus,
    CheckpointListResponse,
    CheckpointInfo
)
from services.training_service import training_service
from services.dataset_service import dataset_service, DatasetStatus
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["Training - Model Fine-tuning"])


@router.post("/start", response_model=TrainingResponse)
async def start_training(request: TrainingStartRequest):
    """
    Start a new training job

    - **dataset_id**: ID of prepared dataset
    - **config**: Training configuration
    - **resume_from**: Optional checkpoint path to resume from

    Dataset must be in 'completed' status (prepared for training).
    """
    # Check if dataset exists and is prepared
    dataset = dataset_service.get_dataset(request.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.status != DatasetStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset is not prepared. Current status: {dataset.status.value}"
        )

    if not dataset.prepared_path:
        raise HTTPException(status_code=400, detail="Dataset has no prepared path")

    # Start training
    config_dict = request.config.model_dump()
    job = training_service.start_training(
        dataset_id=request.dataset_id,
        dataset_path=dataset.prepared_path,
        config=config_dict
    )

    return TrainingResponse(
        success=True,
        job_id=job.job_id,
        status=TrainingStatus(job.status.value),
        checkpoint_path=job.checkpoint_path,
        started_at=job.started_at
    )


@router.get("/jobs", response_model=TrainingListResponse)
async def list_training_jobs():
    """
    List all training jobs
    """
    jobs = training_service.list_jobs()

    return TrainingListResponse(
        success=True,
        jobs=[
            TrainingResponse(
                success=True,
                job_id=job.job_id,
                status=TrainingStatus(job.status.value),
                current_epoch=job.current_epoch,
                current_update=job.current_update,
                total_updates=job.total_updates,
                loss=job.loss,
                learning_rate=job.learning_rate,
                checkpoint_path=job.checkpoint_path,
                error=job.error,
                started_at=job.started_at,
                updated_at=job.updated_at
            )
            for job in jobs
        ],
        total=len(jobs)
    )


@router.get("/jobs/{job_id}", response_model=TrainingResponse)
async def get_training_job(job_id: str):
    """
    Get training job details

    - **job_id**: Training job ID
    """
    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    return TrainingResponse(
        success=True,
        job_id=job.job_id,
        status=TrainingStatus(job.status.value),
        current_epoch=job.current_epoch,
        current_update=job.current_update,
        total_updates=job.total_updates,
        loss=job.loss,
        learning_rate=job.learning_rate,
        checkpoint_path=job.checkpoint_path,
        error=job.error,
        started_at=job.started_at,
        updated_at=job.updated_at
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: str):
    """
    Cancel a training job

    - **job_id**: Training job ID to cancel
    """
    success = training_service.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to cancel job. Job may not exist or is not running."
        )

    return {"success": True, "message": f"Job {job_id} cancelled"}


@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str, delete_checkpoints: bool = False):
    """
    Delete a training job

    - **job_id**: Training job ID to delete
    - **delete_checkpoints**: Also delete checkpoint files
    """
    success = training_service.delete_job(job_id, delete_checkpoints)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to delete job. Job may not exist or is still running."
        )

    return {"success": True, "message": f"Job {job_id} deleted"}


@router.get("/jobs/{job_id}/checkpoints", response_model=CheckpointListResponse)
async def list_checkpoints(job_id: str):
    """
    List checkpoints for a training job

    - **job_id**: Training job ID
    """
    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    checkpoints = training_service.get_checkpoints(job_id)

    return CheckpointListResponse(
        success=True,
        checkpoints=[
            CheckpointInfo(
                name=ckpt["name"],
                path=ckpt["path"],
                update=ckpt["update"],
                size=ckpt["size"],
                created_at=datetime.fromisoformat(ckpt["created_at"])
            )
            for ckpt in checkpoints
        ],
        total=len(checkpoints)
    )


@router.get("/jobs/{job_id}/logs")
async def get_training_logs(job_id: str, lines: int = 100):
    """
    Get training logs for a job

    - **job_id**: Training job ID
    - **lines**: Number of lines to return (from end)
    """
    from pathlib import Path

    job = training_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    log_file = Path(job.checkpoint_path) / "training.log"
    if not log_file.exists():
        return {"success": True, "logs": "", "message": "No logs available yet"}

    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "success": True,
            "logs": "".join(recent_lines),
            "total_lines": len(all_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")


@router.get("/config/defaults")
async def get_default_config():
    """
    Get default training configuration
    """
    return TrainingConfig().model_dump()


@router.get("/stats")
async def get_training_stats():
    """
    Get training statistics
    """
    jobs = training_service.list_jobs()

    total = len(jobs)
    active = sum(1 for j in jobs if j.status.value in ["pending", "preparing", "training"])
    completed = sum(1 for j in jobs if j.status.value == "completed")
    failed = sum(1 for j in jobs if j.status.value == "failed")
    cancelled = sum(1 for j in jobs if j.status.value == "cancelled")

    return {
        "total_jobs": total,
        "active_jobs": active,
        "completed_jobs": completed,
        "failed_jobs": failed,
        "cancelled_jobs": cancelled
    }