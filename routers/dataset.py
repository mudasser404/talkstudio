"""
Dataset Router - Dataset Management API Endpoints
"""

import os
import base64
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks

from models.schemas import (
    DatasetCreateRequest,
    DatasetResponse,
    DatasetListResponse,
    DatasetPrepareRequest,
    DatasetStatus
)
from services.dataset_service import dataset_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["Datasets - Training Data Management"])


@router.post("/create", response_model=DatasetResponse)
async def create_dataset(request: DatasetCreateRequest):
    """
    Create a new dataset

    - **name**: Dataset name
    - **description**: Optional description
    - **language**: Language code (default: en)
    - **is_finetune**: For finetuning (true) or pretraining (false)

    After creation, upload audio files using the upload endpoint.
    """
    dataset = dataset_service.create_dataset(
        name=request.name,
        description=request.description,
        language=request.language,
        is_finetune=request.is_finetune
    )

    return DatasetResponse(
        success=True,
        dataset_id=dataset.dataset_id,
        name=dataset.name,
        status=DatasetStatus(dataset.status.value),
        path=dataset.raw_path
    )


@router.post("/{dataset_id}/upload")
async def upload_audio(
    dataset_id: str,
    audio: UploadFile = File(...),
    text: str = Form(...)
):
    """
    Upload a single audio file to dataset

    - **dataset_id**: Dataset ID
    - **audio**: Audio file (WAV, MP3, FLAC)
    - **text**: Transcript/text for this audio
    """
    audio_data = await audio.read()
    filename = audio.filename or f"audio_{len(audio_data)}.wav"

    success, error = dataset_service.add_audio(
        dataset_id=dataset_id,
        audio_data=audio_data,
        filename=filename,
        text=text
    )

    if not success:
        raise HTTPException(status_code=400, detail=error)

    dataset = dataset_service.get_dataset(dataset_id)

    return {
        "success": True,
        "message": f"Audio uploaded: {filename}",
        "total_samples": dataset.total_samples if dataset else 0,
        "total_duration_hours": dataset.total_duration if dataset else 0
    }


@router.post("/{dataset_id}/upload/batch")
async def upload_audio_batch(
    dataset_id: str,
    audios: List[UploadFile] = File(...),
    texts: str = Form(...)  # JSON array of texts
):
    """
    Upload multiple audio files to dataset

    - **dataset_id**: Dataset ID
    - **audios**: List of audio files
    - **texts**: JSON array of transcripts (same order as audio files)

    Example texts: '["Hello world", "How are you", "Good morning"]'
    """
    import json

    try:
        text_list = json.loads(texts)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in texts parameter")

    if len(audios) != len(text_list):
        raise HTTPException(
            status_code=400,
            detail=f"Number of audio files ({len(audios)}) must match number of texts ({len(text_list)})"
        )

    # Prepare batch data
    audio_files = []
    for audio, text in zip(audios, text_list):
        audio_data = await audio.read()
        filename = audio.filename or f"audio_{len(audio_data)}.wav"
        audio_files.append((audio_data, filename, text))

    success, failed, errors = dataset_service.add_audio_batch(dataset_id, audio_files)

    dataset = dataset_service.get_dataset(dataset_id)

    return {
        "success": failed == 0,
        "uploaded": success,
        "failed": failed,
        "errors": errors[:10] if errors else [],  # Return first 10 errors
        "total_samples": dataset.total_samples if dataset else 0,
        "total_duration_hours": dataset.total_duration if dataset else 0
    }


@router.post("/{dataset_id}/upload/base64")
async def upload_audio_base64(
    dataset_id: str,
    audio_base64: str = Form(...),
    filename: str = Form(...),
    text: str = Form(...)
):
    """
    Upload audio file as base64 string

    - **dataset_id**: Dataset ID
    - **audio_base64**: Base64 encoded audio data
    - **filename**: Filename for the audio
    - **text**: Transcript for this audio
    """
    try:
        audio_data = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    success, error = dataset_service.add_audio(
        dataset_id=dataset_id,
        audio_data=audio_data,
        filename=filename,
        text=text
    )

    if not success:
        raise HTTPException(status_code=400, detail=error)

    return {"success": True, "message": f"Audio uploaded: {filename}"}


@router.post("/{dataset_id}/prepare", response_model=DatasetResponse)
async def prepare_dataset(dataset_id: str, background_tasks: BackgroundTasks):
    """
    Prepare dataset for training

    Converts audio files to arrow format required for training.
    This runs in the background - check status to monitor progress.

    - **dataset_id**: Dataset ID to prepare
    """
    success, error = dataset_service.prepare_dataset(dataset_id)

    if not success:
        raise HTTPException(status_code=400, detail=error)

    dataset = dataset_service.get_dataset(dataset_id)

    return DatasetResponse(
        success=True,
        dataset_id=dataset.dataset_id,
        name=dataset.name,
        status=DatasetStatus(dataset.status.value),
        total_samples=dataset.total_samples,
        total_duration=dataset.total_duration
    )


@router.get("/", response_model=DatasetListResponse)
async def list_datasets():
    """
    List all datasets
    """
    datasets = dataset_service.list_datasets()

    return DatasetListResponse(
        success=True,
        datasets=[
            DatasetResponse(
                success=True,
                dataset_id=ds.dataset_id,
                name=ds.name,
                status=DatasetStatus(ds.status.value),
                total_samples=ds.total_samples,
                total_duration=ds.total_duration,
                path=ds.prepared_path or ds.raw_path,
                error=ds.error
            )
            for ds in datasets
        ],
        total=len(datasets)
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    """
    Get dataset details

    - **dataset_id**: Dataset ID
    """
    dataset = dataset_service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetResponse(
        success=True,
        dataset_id=dataset.dataset_id,
        name=dataset.name,
        status=DatasetStatus(dataset.status.value),
        total_samples=dataset.total_samples,
        total_duration=dataset.total_duration,
        path=dataset.prepared_path or dataset.raw_path,
        error=dataset.error
    )


@router.get("/{dataset_id}/samples")
async def get_dataset_samples(
    dataset_id: str,
    limit: int = 100,
    offset: int = 0
):
    """
    Get samples from dataset

    - **dataset_id**: Dataset ID
    - **limit**: Number of samples to return
    - **offset**: Offset for pagination
    """
    dataset = dataset_service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    samples = dataset_service.get_dataset_samples(dataset_id, limit, offset)

    return {
        "success": True,
        "samples": samples,
        "total": dataset.total_samples,
        "limit": limit,
        "offset": offset
    }


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, delete_files: bool = True):
    """
    Delete a dataset

    - **dataset_id**: Dataset ID to delete
    - **delete_files**: Also delete all associated files
    """
    success = dataset_service.delete_dataset(dataset_id, delete_files)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to delete dataset. It may not exist or is currently being processed."
        )

    return {"success": True, "message": f"Dataset {dataset_id} deleted"}


@router.post("/{dataset_id}/import")
async def import_from_folder(
    dataset_id: str,
    folder_path: str = Form(...),
    metadata_file: Optional[str] = Form(default=None)
):
    """
    Import dataset from existing folder

    The folder should contain:
    - wavs/ directory with audio files
    - metadata.csv with format: audio_file|text

    - **dataset_id**: Dataset ID to import into
    - **folder_path**: Path to folder containing data
    - **metadata_file**: Optional custom metadata filename (default: metadata.csv)
    """
    success, error = dataset_service.import_from_folder(
        dataset_id=dataset_id,
        folder_path=folder_path,
        metadata_file=metadata_file
    )

    if not success:
        raise HTTPException(status_code=400, detail=error)

    dataset = dataset_service.get_dataset(dataset_id)

    return {
        "success": True,
        "message": f"Imported {dataset.total_samples} samples",
        "total_samples": dataset.total_samples,
        "total_duration_hours": dataset.total_duration
    }


@router.get("/stats/summary")
async def get_datasets_summary():
    """
    Get summary statistics for all datasets
    """
    datasets = dataset_service.list_datasets()

    total = len(datasets)
    total_samples = sum(ds.total_samples for ds in datasets)
    total_hours = sum(ds.total_duration for ds in datasets)

    by_status = {}
    for ds in datasets:
        status = ds.status.value
        by_status[status] = by_status.get(status, 0) + 1

    return {
        "total_datasets": total,
        "total_samples": total_samples,
        "total_duration_hours": round(total_hours, 2),
        "by_status": by_status
    }